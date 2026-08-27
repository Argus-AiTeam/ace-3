#!/usr/bin/env python3
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"ace3/model"))
from generation_feedback_compare import ComparisonError,parse_raw,parse_terminal
class Tests(unittest.TestCase):
 def write(self,r,n,t):p=r/n;p.write_text(t,encoding="ascii");return p
 def terminal(self):return "\n".join(["schema=ace3-generation-feedback-terminal-v1","natural_terminal=1","exit_code=0","hidden=4","vocab=5","weights=20","logits=5","top=3","selected=1","feedback=4","commits=1"])+"\n"
 def test_terminal_fail_closed(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t)
   with self.assertRaises(ComparisonError):parse_terminal(self.write(r,"dup",self.terminal()+"feedback=4\n"),0)
   with self.assertRaises(ComparisonError):parse_terminal(self.write(r,"bad",self.terminal().replace("natural_terminal=1","natural_terminal=0")),1)
 def test_malformed_feedback(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);good="selected 0 3c00\nfeedback 0 3c00\nfeedback 1 4000\nfeedback 2 4200\nfeedback 3 4400\ncommit 0 1 "+"11"*32+"\n";self.assertEqual(parse_raw(self.write(r,"good",good),4)["selected"]["token_id"],0)
   for n,b in {"missing":good.replace("selected 0 3c00\n",""),"order":good.replace("feedback 1 4000","feedback 2 4000"),"duplicate":good+"commit 0 1 "+"11"*32+"\n","short":good.replace("feedback 3 4400\n","")}.items():
    with self.subTest(n=n),self.assertRaises(ComparisonError):parse_raw(self.write(r,n,b),4)
 def test_contract(self):
  c=json.loads((ROOT/"ace3/contracts/generated_token_feedback.json").read_text());self.assertFalse(c["source_boundary"]["host_recomputation_allowed"]);self.assertNotIn("ace2",json.dumps(c).lower())
if __name__=="__main__":unittest.main()
