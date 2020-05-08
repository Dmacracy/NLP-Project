export SQUAD_DIR=../data/SQuAD/

python3 run_squad.py \
  --model_type distilbert \
  --model_name_or_path ../models/SQuAD2_trained_longer_model \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/SQuAD-v2.0-train.json \
  --predict_file $SQUAD_DIR/SQuAD-v2.0-dev.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps 10000 \
  --version_2_with_negative \
  --output_dir ../data/Testing/
