import pickle
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import (
    Optional,
    Dict,
)
import numpy as np
from tqdm import tqdm
import logging
#import bitsandbytes as bnb
import pandas as pd

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
    Trainer,
#    BitsAndBytesConfig,
    DataCollatorWithPadding

)
from datasets import (
    load_dataset,
    Dataset
)

from sklearn.metrics import (
    top_k_accuracy_score,
    balanced_accuracy_score,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
import pyarrow
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from sklearn.utils.class_weight import compute_class_weight

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig()
logging.getLogger(__name__).setLevel(logging.INFO)

# for correct loading of parquet data
#https://github.com/huggingface/datasets/issues/6396
pyarrow.PyExtensionType.set_auto_load(True)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[MASK]"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default= "FacebookAI/xlm-roberta-large"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )

@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None, 
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    predict_dataset: str = field(
        default=None,
        metadata={"help": "Which dataset to predict on. See datamodule for options."}
    )
    train_dataset: str = field(
        default='train.parquet',
        metadata={"help": "Which dataset to train on. See datamodule for options."}
    )
    test_dataset: str = field(
      default='test.parquet',
      metadata={"help": "Which dataset to test on. See datamodule for options."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    max_memory_MB: int = field(
        default=32000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='wandb',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    num_labels: int = field(
        default=8,
        metadata={"help":'Number of classes in dataset.'}
    )
    run_name: str = field(
        default='hackathon',
        metadata={"help":'Name of run.'}
    )
    output_dir: str = field(default='./output_cls', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='adamw_torch', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=8, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    #per_device_eval_batch_size: int = field(default=24, metadata={"help":'The evaluation/prediction batch sizer per GPU. Change if out of memory in evaluation/prediction.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=1183, metadata={"help": 'How many optimizer update steps to take.'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=20, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

#define custom Trainer for weighted CE-Loss
class CustomTrainer(Trainer):
    #adapted huggingface impl
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        print(min(labels),max(labels))

        outputs = model(**inputs)

        if isinstance(outputs, dict) and "logits" not in outputs:
            raise ValueError(
                "The model did not return logits from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )

        loss_fct = torch.nn.BCEWithLogitsLoss()
        logits = outputs["logits"] #if isinstance(outputs, dict) else outputs[0]
        print(logits, labels)
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = loss_fct(logits.view(-1,8), labels.view(-1).float())
        return (loss, outputs) if return_outputs else loss

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_accelerate_model(args, checkpoint_dir):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()

    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    print(f"Compute dtype: {compute_dtype}")

    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )

    if args.do_eval or args.do_predict:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            num_labels=args.num_labels,
            #id2label=id2label,
            #label2id=dict((v,k) for k,v in id2label.items()),
            problem_type="multi_label_classification",
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
        )
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'))

            setattr(model, 'model_parallel', True)
            setattr(model, 'is_parallelizable', True)

            model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
        else:
            print("No checkpoints found.")
            return None
    else:

        model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        #load_in_4bit=args.bits == 4,
        #load_in_8bit=args.bits == 8,
        num_labels=args.num_labels,
        #id2label=id2label,
        #label2id=dict((v,k) for k,v in id2label.items()),
        problem_type="multi_label_classification",
        device_map=device_map,
        max_memory=max_memory,
        #quantization_config=BitsAndBytesConfig(
        #    load_in_4bit=args.bits == 4,
        #    load_in_8bit=args.bits == 8,
        #    llm_int8_threshold=6.0,
        #    llm_int8_skip_modules=['classifier'],
        #    llm_int8_has_fp16_weight=False,
        #    bnb_4bit_compute_dtype=compute_dtype,
        #    bnb_4bit_use_double_quant=args.double_quant,
        #    bnb_4bit_quant_type=args.quant_type,
        #),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        )
        setattr(model, 'model_parallel', True)
        setattr(model, 'is_parallelizable', True)
        print(model)
        model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'))

            setattr(model, 'model_parallel', True)
            setattr(model, 'is_parallelizable', True)

            model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
        else:
            print(f'adding Dora modules...')
            config = LoraConfig(
                target_modules='all-linear', #["value","key", "query", "dense"],
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                task_type=TaskType.SEQ_CLS,
                # modules_to_save=['classifier'],
                use_dora=True
            )
            model = get_peft_model(model, config)


    print(model)
    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    """
    tk_train_dataset = None
    tk_eval_dataset = None
    tk_predict_dataset = None
    if args.train_dataset.endswith('.parquet') and args.test_dataset.endswith('.parquet'):
        try:
            #dataset = load_dataset("parquet", data_files={'train': args.train_dataset, 'test': args.test_dataset})
            train_df = pd.read_parquet(args.train_dataset)
            train_dataset = Dataset.from_pandas(train_df)

            test_df = pd.read_parquet(args.test_dataset)
            eval_dataset = Dataset.from_pandas(test_df)

        except:
            raise ValueError(f"Unsupported dataset format: {args.train_dataset}")
    else:
      try:
        dataset = load_dataset(args.dataset)
      except:
        raise ValueError(f"Unsupported dataset format: {args.dataset}. Not able to fetch from huggingface")
    # train data
    if args.do_train:
            try:
                #train_dataset = dataset["train"]
                if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
                    train_dataset = train_dataset.select(range(args.max_train_samples))
            except:
                raise ValueError(f"Error loading train dataset from {args.dataset}")
            tk_train_dataset=train_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True,remove_columns=["text"])
            print(tk_train_dataset)
    # eval data
    if args.do_eval:
            try:
                #eval_dataset = dataset["test"]
                if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
                    eval_dataset = eval_dataset.select(range(args.max_eval_samples))
                tk_eval_dataset = eval_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True, remove_columns=["text"])
            except:
                raise ValueError(f"Error loading eval dataset from {args.dataset}")

    if args.do_predict:
        if args.predict_dataset.endswith('.parquet'):
                try:
                    predict_dataset = load_dataset("parquet", data_files={'predict': args.predict_dataset})
                    tk_predict_dataset = predict_dataset.map(lambda examples: tokenizer(examples["text"],truncation=True), batched=True, remove_columns=["text"])
                except:
                    raise ValueError(f"Error loading dataset from {args.predict_dataset}")
        else:
          raise ValueError(f"Predict set {args.predict_dataset} does not support parquet format.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
        #pad_to_multiple_of=8
    )
    return dict(
        train_dataset=tk_train_dataset if args.do_train else None,
        eval_dataset=tk_eval_dataset if args.do_eval else None,
        predict_dataset=tk_predict_dataset if args.do_predict else None,
        data_collator=data_collator
    )

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        #if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    #metrics
    def compute_metrics(eval_pred):
        print("***Compute Metrics***")
        logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
        predictions = np.argmax(logits, axis=-1)


        class_rep = classification_report(y_true=labels,y_pred=predictions,target_names=id2label.values(),output_dict=True)
        b_accuracy = balanced_accuracy_score(y_true=labels,y_pred=predictions)
        top_2_accuracy = top_k_accuracy_score(y_true=labels,y_score=logits,k=2)
        accuracy = accuracy_score(y_true=labels,y_pred=predictions)
        precision = precision_score(y_true=labels,y_pred=predictions,average="weighted")
        recall = recall_score(y_true=labels,y_pred=predictions,average="weighted")
        f1 = f1_score(y_true=labels,y_pred=predictions, average="weighted")
        #roc_auc = roc_auc_score(y_true=labels,y_score=probabilities,average="weighted")
        # save class report
        with open('class_rep.pickle', 'wb') as handle:
            pickle.dump(class_rep, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(class_rep)
        # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
        return {"precision": precision, "recall": recall, "f1-weighted": f1, 'balanced-accuracy': b_accuracy,"accuracy": accuracy, "top_2_accuracy":top_2_accuracy}#, "roc_auc":roc_auc}

    # define trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
        compute_metrics=compute_metrics
    )

    # Callbacks
    trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    if args.report_to == 'wandb':
        import wandb
        os.environ["WANDB_PROJECT"] = "Hackathon2024"  # name your W&B project
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        wandb.login()



    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logging.info("*** Train ***")
        if completed_training:
            train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
            logging.info("*** Resuming from checkpoint. ***")
        else:
            train_result=trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logging.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        for batch in trainer.get_eval_dataloader(trainer.eval_dataset):
             break
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print(metrics)
        all_metrics.update(metrics)
        if training_args.report_to == "wandb":
            wandb.log(all_metrics)
    # Prediction
    if args.do_predict:
        logging.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        #print(predictions)
        predictions = torch.Tensor(predictions)
        probabilities = torch.softmax(predictions, dim=1).tolist()#[0]
        #print("softmax",probabilities)
        #probabilities = {model.config.id2label[index]: round(probability * 100, 2) for index, probability in enumerate(probabilities)}
        #print(probabilities)
        codes = []
        probs = []
        print("Creating code and probabilities df.")
        for v in probabilities:
            result = {model.config.id2label[index]: round(probability * 100, 2) for index, probability in enumerate(v)}
            result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
            codes.append(list(result.items())[0][0])
            probs.append(list(result.items())[0][1])
        #print("rounded",probabilities)
        #probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        #print(codes, probs)
        # here it is needed to record also confidence values and then cut-off low confidences in analysis later
        # record first item in dict (label+probability value)
        # with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
        #     for i, example in enumerate(data_module['predict_dataset']):
        #         example['prediction_with_input'] = predictions[i].strip()
        #         example['prediction'] = predictions[i].replace(example['input'], '').strip()
        #         fout.write(json.dumps(example) + '\n')
        df = pd.DataFrame({"codes":codes,"probs":probs})
        print(df.head())
        df.to_parquet(args.predict_dataset+"predictions.parquet",compression="zstd")
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)

        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
