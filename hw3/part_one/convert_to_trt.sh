#!/bin/bash
ONNX_MODEL="models/model.onnx"

MIN_BATCH=1
MAX_BATCH=8
SEQ_LEN=128

OPT_BATCH=$(( (MIN_BATCH + MAX_BATCH) / 2 ))

trtexec \
    --onnx=$ONNX_MODEL \
    --minShapes=INPUT_IDS:${MIN_BATCH}x${SEQ_LEN},ATTENTION_MASK:${MIN_BATCH}x${SEQ_LEN} \
    --optShapes=INPUT_IDS:${OPT_BATCH}x${SEQ_LEN},ATTENTION_MASK:${OPT_BATCH}x${SEQ_LEN} \
    --maxShapes=INPUT_IDS:${MAX_BATCH}x${SEQ_LEN},ATTENTION_MASK:${MAX_BATCH}x${SEQ_LEN} \
    --saveEngine=model_fp32.plan

trtexec \
    --onnx=$ONNX_MODEL \
    --minShapes=INPUT_IDS:${MIN_BATCH}x${SEQ_LEN},ATTENTION_MASK:${MIN_BATCH}x${SEQ_LEN} \
    --optShapes=INPUT_IDS:${OPT_BATCH}x${SEQ_LEN},ATTENTION_MASK:${OPT_BATCH}x${SEQ_LEN} \
    --maxShapes=INPUT_IDS:${MAX_BATCH}x${SEQ_LEN},ATTENTION_MASK:${MAX_BATCH}x${SEQ_LEN} \
    --saveEngine=model_fp16.plan \
    --fp16

trtexec \
    --onnx=$ONNX_MODEL \
    --minShapes=INPUT_IDS:${MIN_BATCH}x${SEQ_LEN},ATTENTION_MASK:${MIN_BATCH}x${SEQ_LEN} \
    --optShapes=INPUT_IDS:${OPT_BATCH}x${SEQ_LEN},ATTENTION_MASK:${OPT_BATCH}x${SEQ_LEN} \
    --maxShapes=INPUT_IDS:${MAX_BATCH}x${SEQ_LEN},ATTENTION_MASK:${MAX_BATCH}x${SEQ_LEN} \
    --saveEngine=model_int8.plan \
    --int8

trtexec \
    --onnx=$ONNX_MODEL \
    --minShapes=INPUT_IDS:${MIN_BATCH}x${SEQ_LEN},ATTENTION_MASK:${MIN_BATCH}x${SEQ_LEN} \
    --optShapes=INPUT_IDS:${OPT_BATCH}x${SEQ_LEN},ATTENTION_MASK:${OPT_BATCH}x${SEQ_LEN} \
    --maxShapes=INPUT_IDS:${MAX_BATCH}x${SEQ_LEN},ATTENTION_MASK:${MAX_BATCH}x${SEQ_LEN} \
    --saveEngine=model_best.plan \
    --best
