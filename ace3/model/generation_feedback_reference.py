#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from streaming_lm_head_reference import CHECKPOINT_BYTES,CHECKPOINT_SHA256,HIDDEN_SIZE,MODEL_REPOSITORY,MODEL_REVISION,TOP_K,TIED_WEIGHT_SHA256,VOCAB_SIZE,build_final_rmsnorm_hidden,exact_traversal,sha256_file,tensor_records
TOKENIZER_SHA256="c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
TOKENIZER_CONFIG_SHA256="5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
DOMAIN=b"ace3-generated-token-feedback-v1\0"
def require(ok,message):
    if not ok: raise RuntimeError(message)
def canonical(d): return (json.dumps(d,sort_keys=True,separators=(",",":"),ensure_ascii=True)+"\n").encode("ascii")
def lines(p,values): p.write_text("".join(f"{v}\n" for v in values),encoding="ascii")
def generate(checkpoint,tokenizer_dir,output_dir):
    require(checkpoint.stat().st_size==CHECKPOINT_BYTES,"checkpoint byte count mismatch");require(sha256_file(checkpoint)==CHECKPOINT_SHA256,"checkpoint SHA256 mismatch")
    tp=tokenizer_dir/"tokenizer.json";cp=tokenizer_dir/"tokenizer_config.json"
    require(sha256_file(tp)==TOKENIZER_SHA256,"tokenizer SHA256 mismatch");require(sha256_file(cp)==TOKENIZER_CONFIG_SHA256,"tokenizer config SHA256 mismatch")
    records=tensor_records(checkpoint);hidden=build_final_rmsnorm_hidden(checkpoint,records);winners,_,logits_sha=exact_traversal(checkpoint,records,hidden)
    token,logit,q24=winners[0];embedding=records["model.embed_tokens.weight"];row_offset=embedding["offset"]+token*HIDDEN_SIZE*2
    with checkpoint.open("rb") as f:f.seek(row_offset);payload=f.read(HIDDEN_SIZE*2)
    require(len(payload)==HIDDEN_SIZE*2,"embedding row truncated");bits=np.frombuffer(payload,dtype="<u2")
    tokenizer=Tokenizer.from_file(str(tp));token_text=tokenizer.id_to_token(token);require(token_text is not None,"token absent from tokenizer")
    decoded=tokenizer.decode([token],skip_special_tokens=False);td=json.loads(tp.read_bytes());special={int(x["id"]) for x in td.get("added_tokens",[]) if x.get("special")}
    hidden_payload=np.asarray(hidden,dtype="<u2").tobytes();prior_doc={"checkpoint_sha256":CHECKPOINT_SHA256,"domain":"accepted-final-rmsnorm-to-streaming-lm-head","hidden_sha256":hashlib.sha256(hidden_payload).hexdigest(),"position":0}
    prior=hashlib.sha256(canonical(prior_doc)).digest();next_tip=hashlib.sha256(DOMAIN+prior+struct.pack("<I",token)+struct.pack("<I",1)+payload).hexdigest()
    output_dir.mkdir(parents=True,exist_ok=True);lines(output_dir/"hidden.hex",(f"{int(x):04x}" for x in hidden));lines(output_dir/"oracle_embedding.hex",(f"{int(x):04x}" for x in bits))
    cfg={"checkpoint_bytes":CHECKPOINT_BYTES,"weight_offset":embedding["offset"],"weight_bytes":embedding["bytes"],"hidden_size":HIDDEN_SIZE,"vocab_size":VOCAB_SIZE,"top_k":TOP_K,"checkpoint_sha256":CHECKPOINT_SHA256,"vocabulary_sha256":TOKENIZER_SHA256,"tokenizer_config_sha256":TOKENIZER_CONFIG_SHA256,"trusted_state_tip":prior.hex(),"presented_state_tip":prior.hex(),"prior_position":0,"next_position":1}
    lines(output_dir/"input.cfg",(f"{k}={v}" for k,v in cfg.items()))
    oracle={"schema":"ace3-generation-feedback-oracle-v1","model":{"repository":MODEL_REPOSITORY,"revision":MODEL_REVISION,"checkpoint_sha256":CHECKPOINT_SHA256,"tied_value_sha256":TIED_WEIGHT_SHA256},"geometry":{"hidden_size":HIDDEN_SIZE,"vocab_size":VOCAB_SIZE,"top_k":TOP_K},"hidden_sha256":hashlib.sha256(hidden_payload).hexdigest(),"logits_sha256":logits_sha,"selection_policy":"descending rounded finite FP16 logit, then ascending token ID","top_k":[{"rank":r,"token_id":t,"logit_f16_bits":b,"logit_q24":v} for r,(t,b,v) in enumerate(winners)],"selected":{"token_id":token,"logit_f16_bits":logit,"logit_q24":q24},"decode":{"token_id":token,"tokenizer_token":token_text,"decoded_text":decoded,"decoded_utf8_hex":decoded.encode().hex(),"special":token in special,"tokenizer_sha256":TOKENIZER_SHA256,"tokenizer_config_sha256":TOKENIZER_CONFIG_SHA256},"embedding":{"tensor":"model.embed_tokens.weight","token_id":token,"elements":HIDDEN_SIZE,"sha256":hashlib.sha256(payload).hexdigest()},"state":{"prior_position":0,"next_position":1,"prior_state_tip":prior.hex(),"next_state_tip":next_tip}}
    (output_dir/"oracle.json").write_text(json.dumps(oracle,indent=2,sort_keys=True)+"\n")
    manifest={"schema":"ace3-generation-feedback-inputs-v1","checkpoint":{"path_recorded":False,"sha256":CHECKPOINT_SHA256,"bytes":CHECKPOINT_BYTES},"tokenizer":{"path_recorded":False,"sha256":TOKENIZER_SHA256,"config_sha256":TOKENIZER_CONFIG_SHA256},"artifacts":{n:hashlib.sha256((output_dir/n).read_bytes()).hexdigest() for n in ("hidden.hex","input.cfg","oracle_embedding.hex","oracle.json")},"ace2_dependency":False}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");return oracle
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--tokenizer-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();o=generate(a.checkpoint,a.tokenizer_dir,a.output_dir);print(f"GENERATION_FEEDBACK_REFERENCE_PASS token={o['selected']['token_id']} embedding=896 decode_utf8={o['decode']['decoded_utf8_hex']}")
if __name__=="__main__":main()
