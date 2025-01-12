### 模型代码
class SenseVoiceSmall(nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        ctc_conf: dict = None,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        length_normalized_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)

        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.error_calculator = None

        self.ctc = ctc

        self.length_normalized_loss = length_normalized_loss
        self.encoder_output_size = encoder_output_size

        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.embed = torch.nn.Embedding(
            7 + len(self.lid_dict) + len(self.textnorm_dict), input_size
        )
        self.emo_dict = {
            "unk": 25009,
            "happy": 25001,
            "sad": 25002,
            "angry": 25003,
            "neutral": 25004,
        }

        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )

    @staticmethod
    def from_pretrained(model: str = None, **kwargs):
        from funasr import AutoModel

        model, kwargs = AutoModel.build_model(model=model, trust_remote_code=True, **kwargs)

        return model, kwargs

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, text)

        loss_ctc, cer_ctc = None, None
        loss_rich, acc_rich = None, None
        stats = dict()

        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out[:, 4:, :], encoder_out_lens - 4, text[:, 4:], text_lengths - 4
        )

        loss_rich, acc_rich = self._calc_rich_ce_loss(encoder_out[:, :4, :], text[:, :4])

        loss = loss_ctc + loss_rich
        # Collect total loss stats
        stats["loss_ctc"] = torch.clone(loss_ctc.detach()) if loss_ctc is not None else None
        stats["loss_rich"] = torch.clone(loss_rich.detach()) if loss_rich is not None else None
        stats["loss"] = torch.clone(loss.detach()) if loss is not None else None
        stats["acc_rich"] = acc_rich

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        **kwargs,
    ):
        # Data augmentation
        if self.specaug is not None and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)

        # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        lids = torch.LongTensor(
            [
                [
                    (
                        self.lid_int_dict[int(lid)]
                        if torch.rand(1) > 0.2 and int(lid) in self.lid_int_dict
                        else 0
                    )
                ]
                for lid in text[:, 0]
            ]
        ).to(speech.device)
        language_query = self.embed(lids)

        styles = torch.LongTensor(
            [[self.textnorm_int_dict[int(style)]] for style in text[:, 3]]
        ).to(speech.device)
        style_query = self.embed(styles)
        speech = torch.cat((style_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)

        return encoder_out, encoder_out_lens

### 训练代码
@hydra.main(config_name=None, version_base=None)
def main_hydra(kwargs: DictConfig):
    main(**kwargs)

def main(**kwargs):
    model = AutoModel(**kwargs)

    # save config.yaml
    if rank == 0:
        prepare_model_dir(**kwargs)

    # parse kwargs
    kwargs = model.kwargs
    kwargs["device"] = device
    tokenizer = kwargs["tokenizer"]
    frontend = kwargs["frontend"]
    model = model.model
    del kwargs["model"]

    trainer = Trainer(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        use_ddp=use_ddp,
        use_fsdp=use_fsdp,
        device=kwargs["device"],
        excludes=kwargs.get("excludes", None),
        output_dir=kwargs.get("output_dir", "./exp"),
        **kwargs.get("train_conf"),
    )

    model = trainer.warp_model(model, **kwargs)

    model, optim, scheduler = trainer.warp_optim_scheduler(model, **kwargs)

    # dataset
    logging.info("Build dataloader")
    dataloader_class = DataloaderMapStyle()
    dataloader = dataloader_class(**kwargs)
    # dataloader_tr, dataloader_val = dataloader_class(**kwargs)

    scaler = GradScaler(enabled=True) if trainer.use_fp16 or trainer.use_bf16 else None
    scaler = ShardedGradScaler(enabled=trainer.use_fp16) if trainer.use_fsdp else scaler

    dataloader_tr, dataloader_val = None, None
    for epoch in range(trainer.start_epoch, trainer.max_epoch):
        time1 = time.perf_counter()

        for data_split_i in range(trainer.start_data_split_i, dataloader.data_split_num):
            time_slice_i = time.perf_counter()

            dataloader_tr, dataloader_val = dataloader.build_iter(
                epoch, data_split_i=data_split_i, start_step=trainer.start_step
            )

            trainer.train_epoch(
                model=model,
                optim=optim,
                scheduler=scheduler,
                scaler=scaler,
                dataloader_train=dataloader_tr,
                dataloader_val=dataloader_val,
                epoch=epoch,
                data_split_i=data_split_i,
                data_split_num=dataloader.data_split_num,
                start_step=trainer.start_step,
            )
            trainer.start_step = 0

            torch.cuda.empty_cache()

            time_escaped = (time.perf_counter() - time_slice_i) / 3600.0

        trainer.start_data_split_i = 0
        trainer.validate_epoch(model=model, dataloader_val=dataloader_val, epoch=epoch + 1)
        scheduler.step()
        trainer.step_in_epoch = 0
        trainer.save_checkpoint(
            epoch + 1, model=model, optim=optim, scheduler=scheduler, scaler=scaler
        )

        time2 = time.perf_counter()
        time_escaped = (time2 - time1) / 3600.0

        trainer.train_acc_avg = 0.0
        trainer.train_loss_avg = 0.0

    trainer.close()


if __name__ == "__main__":
    main_hydra()

### dataset代码
def DataloaderMapStyle(frontend=None, tokenizer=None, **kwargs):
    # dataset
    logging.info("Build dataloader")
    dataset_class = SenseVoiceCTCDataset()
    dataset_tr = dataset_class(
        kwargs.get("train_data_set_list"),
        frontend=frontend,
        tokenizer=tokenizer,
        is_training=True,
        **kwargs.get("dataset_conf"),
    )
    dataset_val = dataset_class(
        kwargs.get("valid_data_set_list"),
        frontend=frontend,
        tokenizer=tokenizer,
        is_training=False,
        **kwargs.get("dataset_conf"),
    )

    # dataloader
    batch_sampler = kwargs["dataset_conf"].get("batch_sampler", "BatchSampler")
    batch_sampler_val = None
    if batch_sampler is not None:
        batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
        batch_sampler = batch_sampler_class(dataset_tr, **kwargs.get("dataset_conf"))
        batch_sampler_val = batch_sampler_class(
            dataset_val, is_training=False, **kwargs.get("dataset_conf")
        )

    dataloader_tr = torch.utils.data.DataLoader(
        dataset_tr, collate_fn=dataset_tr.collator, **batch_sampler
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, collate_fn=dataset_val.collator, **batch_sampler_val
    )

    return dataloader_tr, dataloader_val


class SenseVoiceCTCDataset(torch.utils.data.Dataset):
    """
    SenseVoiceCTCDataset
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        index_ds_class = IndexDSJsonlRankFull()
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 5)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

    def __getitem__(self, index):

        output = None
        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            source = item["source"]
            try:
                data_src = load_audio_text_image_video(source, fs=self.fs)
            except Exception as e:
                logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")
                continue

            if self.preprocessor_speech:
                data_src = self.preprocessor_speech(data_src, fs=self.fs)
            speech, speech_lengths = extract_fbank(
                data_src, data_type=self.data_type, frontend=self.frontend, is_final=True
            )  # speech: [b, T, d]

            if speech_lengths > self.batch_size:
                continue
            if self.permute:
                speech = speech.permute(0, 2, 1)
            asr_target = item["target"]
            if self.preprocessor_text:
                asr_target = self.preprocessor_text(asr_target)
            emo_target = item.get("emo_target", "<|NEUTRAL|>")
            event_target = item.get("event_target", "<|Speech|>")
            text_language = item.get("text_language", "<|zh|>")
            punc_itn_bottom = item.get("with_or_wo_itn", "<|woitn|>")

            target_ids = self.tokenizer.encode(asr_target, allowed_special="all")
            target_ids_len = len(target_ids)  # [text]
            if target_ids_len > 200:
                continue

            lid_ids = self.tokenizer.encode(text_language, allowed_special="all")
            emo_ids = self.tokenizer.encode(emo_target, allowed_special="all")
            event_ids = self.tokenizer.encode(event_target, allowed_special="all")
            punc_itn_bottom_ids = self.tokenizer.encode(punc_itn_bottom, allowed_special="all")

            ids = lid_ids + emo_ids + event_ids + punc_itn_bottom_ids + target_ids # [lid, emo, lid, itn, text]
            ids_lengths = len(ids)

            text = torch.tensor(ids, dtype=torch.int64)
            text_lengths = torch.tensor([ids_lengths], dtype=torch.int32)

            output = {
                "speech": speech[0, :, :],
                "speech_lengths": speech_lengths,
                "text": text,
                "text_lengths": text_lengths,
            }
            break

        return output

    def collator(self, samples: list = None):
        outputs = {}
        for sample in samples:
            if sample is None:
                continue
            for key in sample.keys():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(sample[key])

        if len(outputs) < 1:
            logging.error(f"ERROR: data is empty!")
            outputs = {
                "speech": torch.rand((10, 128), dtype=torch.float32)[None, :, :],
                "speech_lengths": torch.tensor(
                    [
                        10,
                    ],
                    dtype=torch.int32,
                )[:, None],
                "text": torch.tensor(
                    [
                        58836,
                    ],
                    dtype=torch.int32,
                )[None, :],
                "text_lengths": torch.tensor(
                    [
                        1,
                    ],
                    dtype=torch.int32,
                )[:, None],
            }
            return outputs

        for key, data_list in outputs.items():
            if isinstance(data_list[0], torch.Tensor):
                if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                    pad_value = self.int_pad_value
                else:
                    pad_value = self.float_pad_value

                outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )

        if self.batch_type != "example":
            for i in range(10):
                outputs = self._filter_badcase(outputs, i=i)

        return outputs

### Index代码
class IndexDSJsonlRankFull(torch.utils.data.Dataset):

    def __init__(self, path: str, **kwargs):
        super().__init__()
        self.max_source_length = kwargs.get("max_source_length", 2048)
        self.min_source_length = kwargs.get("min_source_length", 0)
        self.max_target_length = kwargs.get("max_target_length", 2048)
        self.min_target_length = kwargs.get("min_target_length", 0)
        self.max_token_length = kwargs.get("max_token_length", 2200)

        is_training = kwargs.get("is_training", True)
        if not (path.endswith(".jsonl") or path.endswith(".json")):
            # jsonl list file
            data_split_num = kwargs.get("data_split_num", 1)
            data_split_i = kwargs.get("data_split_i", 0)

            if not is_training:
                data_split_num = 1
                data_split_i = 0
            with open(path, encoding="utf-8") as fin:
                file_list_all = fin.readlines()

                num_per_slice = (len(file_list_all) - 1) // data_split_num + 1  # 16
                file_list = file_list_all[
                    data_split_i * num_per_slice : (data_split_i + 1) * num_per_slice
                ]
                logging.info(
                    f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list: {file_list}, \nfile_list_all: {file_list_all}"
                )

        else:
            file_list = [path]

        contents = []
        for file_json in file_list:
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    data = json.loads(line.strip())
                    if "text" in data:  # for sft
                        contents.append(data["text"])
                    if "source" in data:  # for speech lab pretrain
                        prompt = data.get("prompt", "<ASR>")
                        source = data["source"].replace(
                            "/cpfs01", "/cpfs_speech/data"
                        )  # only use in alibaba gpu group: .replace("/cpfs01", "/cpfs_speech/data")
                        target = data["target"]
                        source_len = data.get("source_len", 1)
                        target_len = data.get("target_len", 0)
                        if "aishell" in source:
                            target = target.replace(" ", "")
                        if (
                            source_len < self.min_source_length
                            or source_len > self.max_source_length
                        ):
                            continue
                        if (
                            target_len < self.min_target_length
                            or target_len > self.max_target_length
                        ):
                            continue

                        if (source_len + target_len) > self.max_token_length:
                            continue

                        contents_i = {
                            "source": source,
                            "prompt": prompt,
                            "target": target,
                            "source_len": source_len,
                            "target_len": target_len,
                        }
                        text_language = data.get("text_language", None)
                        if text_language is not None:
                            contents_i["text_language"] = text_language
                        if "emo_target" in data:
                            contents_i["emo_target"] = data["emo_target"]
                        if "event_target" in data:
                            contents_i["event_target"] = data["event_target"]
                        if "with_or_wo_itn" in data:
                            contents_i["with_or_wo_itn"] = data["with_or_wo_itn"]
                        # audio_language = data.get("audio_language", None)
                        # if audio_language is not None:
                        #     contents_i["audio_language"] = audio_language
                        contents.append(contents_i)

        self.contents = contents

        logging.info("total_num of samplers: {}, {}".format(len(self.contents), path))

### 训练数据
{"key": "YOU0000008470_S0000238_punc_itn", "text_language": "<|en|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|withitn|>", "target": "Including legal due diligence, subscription agreement, negotiation.", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/industrial_data/english_all/audio/YOU0000008470_S0000238.wav", "target_len": 7, "source_len": 140, "speaker": "xiaoming"}
{"key": "AUD0000001556_S0007580", "text_language": "<|en|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|woitn|>", "target": "there is a tendency to identify the self or take interest in what one has got used to", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/industrial_data/english_all/audio/AUD0000001556_S0007580.wav", "target_len": 18, "source_len": 360, "speaker": "<|Unknown_Speaker|>"}

以上是一个模型的实现及训练代码，请调整代码使得模型在保留原有能力的同时，能够训练输出较高质量的speaker embedding，要求：
1 应用State Pooling Layer 提取speaker embedding
2 训练数据中speaker标签可能是真实的值，也可能是<|Unknown_Speaker|>
3 通过说话人分类任务进行speaker embedding的训练，speaker numbers从训练数据中预先统计
4 请保证训练时每个batch一定有一条数据的speak label不为<|Unknown_Speaker|>
5 请在回答中省略未改动的代码并对新增或改动的代码给出说明
6 请你仔细慢慢思考后 再给出完整详细的回答
