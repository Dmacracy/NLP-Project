export SQUAD_DIR=../data/SQuAD/

python3 run_squad.py \
  --model_type distilbert \
  --model_name_or_path distilbert-base-uncased-distilled-squad \
  --do_train \
  --do_eval \
  --train_file $SQUAD_DIR/SQuAD-v2.0-train.json \
  --predict_file $SQUAD_DIR/SQuAD-v2.0-dev.json \
  --per_gpu_train_batch_size 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ../data/SQuAD_out/
