### 训练数据样例
{"key": "YOU0000008470_S0000238_punc_itn", "text_language": "<|en|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|withitn|>", "target": "Including legal due diligence, subscription agreement, negotiation.", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/industrial_data/english_all/audio/YOU0000008470_S0000238.wav", "target_len": 7, "source_len": 140, "speaker": "xiaoming"}
{"key": "AUD0000001556_S0007580", "text_language": "<|en|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|woitn|>", "target": "there is a tendency to identify the self or take interest in what one has got used to", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/industrial_data/english_all/audio/AUD0000001556_S0007580.wav", "target_len": 18, "source_len": 360, "speaker": "<|Unknown_Speaker|>"}

### 训练代码

def main(**kwargs):

    model = AutoModel(**kwargs)
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

    kwargs["device"] = int(os.environ.get("LOCAL_RANK", 0))
    trainer.device = int(os.environ.get("LOCAL_RANK", 0))

    model, optim, scheduler = trainer.warp_optim_scheduler(model, **kwargs)

    # dataset
    logging.info("Build dataloader")
    dataloader_class = DataloaderMapStyle()
    dataloader = dataloader_class(**kwargs)
 
    scaler = GradScaler(enabled=True) if trainer.use_fp16 or trainer.use_bf16 else None
    scaler = ShardedGradScaler(enabled=trainer.use_fp16) if trainer.use_fsdp else scaler

    trainer.resume_checkpoint(
        model=model,
        optim=optim,
        scheduler=scheduler,
        scaler=scaler,
    )

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

    if trainer.rank == 0:
        average_checkpoints(
            trainer.output_dir, trainer.avg_nbest_model, use_deepspeed=trainer.use_deepspeed
        )

    trainer.close()


if __name__ == "__main__":
    main_hydra()

### Dataloader代码
class DataloaderMapStyle:
    def __init__(self, frontend=None, tokenizer=None, **kwargs):
        # dataset
        logging.info("Build dataloader")

        dataset_class = SenseVoiceCTCDataset()
        dataset_tr = None
        # split dataset
        self.data_split_num = kwargs["dataset_conf"].get("data_split_num", 1)
        if self.data_split_num == 1:
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

        self.dataset_tr = dataset_tr
        self.dataset_val = dataset_val
        self.kwargs = kwargs

        self.dataset_class = dataset_class
        self.frontend = frontend
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def build_iter(self, epoch=0, data_split_i=0, start_step=0, **kwargs):

        # dataloader
        batch_sampler = self.kwargs["dataset_conf"].get("batch_sampler", "BatchSampler")
        batch_sampler_val = None
        if batch_sampler is not None:
            batch_sampler = CustomDistributedBatchSampler_fn(
                self.dataset_tr, start_step=start_step, **self.kwargs.get("dataset_conf")
            )
            batch_sampler_val = batch_sampler_class(
                self.dataset_val, is_training=False, **self.kwargs.get("dataset_conf")
            )

        batch_sampler["batch_sampler"].set_epoch(epoch)
        batch_sampler_val["batch_sampler"].set_epoch(epoch)
        dataloader_tr = torch.utils.data.DataLoader(
            self.dataset_tr, collate_fn=self.dataset_tr.collator, **batch_sampler
        )
        dataloader_val = torch.utils.data.DataLoader(
            self.dataset_val, collate_fn=self.dataset_val.collator, **batch_sampler_val
        )

        return dataloader_tr, dataloader_val

### sampler
def CustomDistributedBatchSampler_fn(dataset, **kwargs):
    dataloader_args = {}
    batch_sampler = CustomDistributedBufferDynamicBatchSampler(dataset, **kwargs)

    dataloader_args["batch_sampler"] = batch_sampler
    dataloader_args["num_workers"] = kwargs.get("num_workers", 4)
    dataloader_args["pin_memory"] = kwargs.get("pin_memory", True)

    return dataloader_args

class CustomDistributedBufferDynamicBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        batch_type="token",
        num_replicas=None,
        rank=None,
        rank_split=False,
        shuffle=True,
        drop_last=False,
        is_training: bool = True,
        sort_size: int = 1024,
        start_step: int = 0,
        **kwargs,
    ):

        try:
            rank = dist.get_rank()
            num_replicas = dist.get_world_size()
        except:
            rank = 0
            num_replicas = 1

        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.is_training = is_training
        self.shuffle = shuffle and is_training
        self.drop_last = drop_last

        self.total_size = len(self.dataset)
        self.num_samples = int(math.ceil(self.total_size / self.num_replicas))
        self.epoch = 0
        self.sort_size = sort_size * num_replicas
        self.max_token_length = kwargs.get("max_token_length", 2048)
        self.length_scale_source = kwargs.get("length_scale_source", 1.0)
        self.batch_size_sample_max = kwargs.get("batch_size_sample_max", 200)
        self.start_step = start_step
        self.batch_num = 1
        if self.start_step > 0:
            logging.info(f"Warning, start_step > 0, dataloader start from step: {self.start_step}")

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            random.seed(self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Create sorted buffers and form batches
        buffer_batches = []
        for i in range(0, len(indices), self.sort_size):
            buffer = sorted(
                indices[i : i + self.sort_size], key=lambda idx: self.dataset.get_source_len(idx)
            )
            batch = []
            max_len_in_batch = 0
            count = 1
            for idx in buffer:
                original_sample_length = self.dataset.get_source_len(idx)
                if original_sample_length > self.max_token_length:
                    continue
                sample_length = 1 if self.batch_type == "example" else original_sample_length
                potential_batch_length = max(max_len_in_batch, sample_length) * (len(batch) + 1)
                if potential_batch_length <= self.batch_size and count < self.batch_size_sample_max:
                    batch.append(idx)
                    max_len_in_batch = max(max_len_in_batch, sample_length)
                    count += 1
                else:
                    buffer_batches.append(batch)
                    batch = [idx]
                    max_len_in_batch = sample_length
                    count = 1
            if batch:
                buffer_batches.append(batch)

        # Ensure each rank gets the same number of batches, duplicate data if needed
        batches_per_rank = math.ceil(len(buffer_batches) / self.num_replicas)
        total_batches_needed = batches_per_rank * self.num_replicas

        extra_batches = total_batches_needed - len(buffer_batches)
        buffer_batches += random.choices(buffer_batches, k=extra_batches)

        # Evenly distribute batches from buffer_batches to each rank
        rank_batches = [[] for _ in range(self.num_replicas)]
        for i, batch in enumerate(buffer_batches):
            rank_batches[i % self.num_replicas].append(batch)

        # Assign all batches for the current rank directly
        final_batches = rank_batches[self.rank][self.start_step :]
        self.batch_num = len(final_batches)

        logging.info(
            f"rank: {self.rank}, dataloader start from step: {self.start_step}, batch_num: {len(rank_batches[self.rank])}, after: {self.batch_num}"
        )
        return iter(final_batches)

### dataset代码
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

            if speech_lengths > self.batch_size:
                continue
            if self.permute:
                speech = speech.permute(0, 2, 1)
            asr_target = item["target"]

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

        return outputs

###index代码

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

                        target = data["target"]
                        source_len = data.get("source_len", 1)
                        target_len = data.get("target_len", 0)

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
                        contents.append(contents_i)

        self.contents = contents

        logging.info("total_num of samplers: {}, {}".format(len(self.contents), path))

###model代码
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

        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if ctc_conf is None:
            ctc_conf = {}
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

###任务
以上是一个模型的实现及训练代码，请调整代码使得模型在保留原有能力的同时，能够训练输出较高质量的speaker embedding，要求：
1 应用State Pooling Layer 提取speaker embedding
2 训练数据中speaker label可能是真实的人名，也可能是<|Unknown_Speaker|>
3 通过说话人分类任务进行speaker embedding的训练，speaker numbers从训练数据中预先统计
4 训练时如果batch中所有数据的speak label均为<|Unknown_Speaker|>，则随机抽取一条speaker label不为<|Unknown_Speaker|>的数据添加到batch中
5 请在回答中省略未改动的代码并对新增或改动的代码给出说明
6 请你仔细慢慢思考后 再给出完整详细的回答
