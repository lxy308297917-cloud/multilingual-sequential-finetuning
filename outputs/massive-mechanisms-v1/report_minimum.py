"""Apply the identical paired analysis to the supplemental minimum dose."""
from pathlib import Path

source=Path(__file__).with_name("report.py")
code=source.read_text().replace('out/"mechanism"','out/"mechanism-minimum"')
code=code.replace('arms+=([baseline] if shared else [])','arms.extend(out/"mechanism"/k for k in ("P0","P1"))')
code=code.replace('read(out/"A_SELECTION.json")["selected"]','{k: k+"_0.125" for k in ("functional","uniform","random","old_loss","format")}')
code=code.replace('out/"RESULTS.json"','out/"A_MINIMUM_RESULTS.json"').replace('out/"SCORES.csv"','out/"A_MINIMUM_SCORES.csv"')
exec(compile(code,str(source),"exec"),{"__file__":str(source),"__name__":"__main__"})
