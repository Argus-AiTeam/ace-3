#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,struct
from pathlib import Path
SCHEMA="ace3-generation-feedback-terminal-v1";DOMAIN=b"ace3-generated-token-feedback-v1\0"
class ComparisonError(RuntimeError):pass
def require(ok,msg):
    if not ok:raise ComparisonError(msg)
def ascii_text(p):
    try:return p.read_text(encoding="ascii")
    except (UnicodeDecodeError,OSError) as e:raise ComparisonError(f"cannot read ASCII {p.name}: {e}") from e
def parse_terminal(path,actual):
    r={}
    for line in ascii_text(path).splitlines():
        require(line.count("=")==1,"malformed terminal field");k,v=line.split("=",1);require(k and v and k not in r and k.isascii() and v.isascii(),"duplicate or invalid terminal field");r[k]=v
    expected={"schema","natural_terminal","exit_code","hidden","vocab","weights","logits","top","selected","feedback","commits"};require(set(r)==expected,"terminal fields mismatch");require(r["schema"]==SCHEMA,"terminal schema mismatch");require(r["natural_terminal"]=="1","simulator did not reach natural terminal");require(int(r["exit_code"])==actual==0,"simulator exit mismatch");return r
def parse_raw(path,hidden_size):
    selected=None;commit=None;feedback=[]
    for line in ascii_text(path).splitlines():
        f=line.split();require(f,"empty raw row")
        if f[0]=="selected":require(len(f)==3 and selected is None,"duplicate or malformed selected row");selected={"token_id":int(f[1]),"logit_f16_bits":int(f[2],16)}
        elif f[0]=="feedback":require(len(f)==3 and int(f[1])==len(feedback),"malformed feedback order");feedback.append(int(f[2],16))
        elif f[0]=="commit":require(len(f)==4 and commit is None,"duplicate or malformed commit row");commit={"token_id":int(f[1]),"position":int(f[2]),"prior_state_tip":f[3]}
        else:raise ComparisonError("unknown raw row")
    require(selected is not None,"missing selected row");require(len(feedback)==hidden_size,"feedback vector length mismatch");require(commit is not None,"missing commit row");return {"selected":selected,"feedback":feedback,"commit":commit}
def compare(terminal,actual,raw_path,oracle_path,embedding_path,report_path):
    term=parse_terminal(terminal,actual);oracle=json.loads(oracle_path.read_text(encoding="ascii"));require(oracle.get("schema")=="ace3-generation-feedback-oracle-v1","oracle schema mismatch")
    h=int(oracle["geometry"]["hidden_size"]);raw=parse_raw(raw_path,h);expected=oracle["selected"];require(raw["selected"]=={"token_id":expected["token_id"],"logit_f16_bits":expected["logit_f16_bits"]},"RTL selected token/logit mismatch")
    embedding=[int(x,16) for x in ascii_text(embedding_path).splitlines()];require(raw["feedback"]==embedding,"embedding feedback vector mismatch");commit=raw["commit"];state=oracle["state"]
    require(commit["token_id"]==expected["token_id"],"commit token mismatch");require(commit["position"]==state["next_position"],"commit position mismatch");require(commit["prior_state_tip"]==state["prior_state_tip"],"commit state tip mismatch")
    payload=b"".join(struct.pack("<H",x) for x in raw["feedback"]);tip=hashlib.sha256(DOMAIN+bytes.fromhex(commit["prior_state_tip"])+struct.pack("<I",commit["token_id"])+struct.pack("<I",commit["position"])+payload).hexdigest();require(tip==state["next_state_tip"],"next causal state tip mismatch")
    require(int(term["hidden"])==h and int(term["vocab"])==oracle["geometry"]["vocab_size"] and int(term["weights"])==h*oracle["geometry"]["vocab_size"] and int(term["logits"])==oracle["geometry"]["vocab_size"] and int(term["top"])==oracle["geometry"]["top_k"] and int(term["selected"])==1 and int(term["feedback"])==h and int(term["commits"])==1,"terminal counts mismatch")
    report={"schema":"ace3-generation-feedback-comparison-v1","pass":True,"rtl_selected_token_id":commit["token_id"],"exact_embedding_words":h,"decode":oracle["decode"],"prior_state_tip":commit["prior_state_tip"],"next_state_tip":tip,"next_position":commit["position"],"selection_source":"raw accepted RTL rank-zero output","oracle_role":"post-terminal comparison only"};report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="ascii");return report
def main():
    p=argparse.ArgumentParser();p.add_argument("--terminal",type=Path,required=True);p.add_argument("--exit-code",type=Path,required=True);p.add_argument("--raw",type=Path,required=True);p.add_argument("--oracle",type=Path,required=True);p.add_argument("--embedding",type=Path,required=True);p.add_argument("--report",type=Path,required=True);a=p.parse_args();r=compare(a.terminal,int(ascii_text(a.exit_code).strip()),a.raw,a.oracle,a.embedding,a.report);print(f"GENERATION_FEEDBACK_COMPARE_PASS token={r['rtl_selected_token_id']} embedding={r['exact_embedding_words']} next_position={r['next_position']}")
if __name__=="__main__":main()
