# C:\Users\melik\AQRE\tools\generate_mock_data_v2.py
# -*- coding: utf-8 -*-
import sys; sys.path.append(r"C:\Users\melik\AQRE")
from src.config import RAW_DATA_DIR
from pathlib import Path
import numpy as np, pandas as pd

RAW = RAW_DATA_DIR
N_TOTAL = 258
SEED = 42
MARGIN = 1.06
DATE_START = pd.Timestamp("2024-08-01")
DATE_END   = pd.Timestamp("2025-05-31")
rng = np.random.default_rng(SEED)

def ensure_dirs(): RAW.mkdir(parents=True, exist_ok=True)
def make_calendar(n:int)->pd.Series:
    days=max((DATE_END-DATE_START).days,1)
    offs=rng.integers(0,days,size=n); hrs=rng.choice([13,15,17,19,21],size=n,p=[.1,.15,.3,.3,.15])
    dt=(DATE_START+pd.to_timedelta(offs,unit="D"))+pd.to_timedelta(hrs,unit="h")
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def softmax_rows(a):
    a=a-a.max(axis=1,keepdims=True); e=np.exp(a); return e/e.sum(axis=1,keepdims=True)
def generate_probabilities(n):
    base=np.c_[rng.normal(.35,.10,n),rng.normal(.20,.07,n),rng.normal(.30,.10,n)]
    prob=softmax_rows(base+rng.normal(0,.05,(n,3))); prob=np.clip(prob,1e-4,1); return prob/prob.sum(axis=1,keepdims=True)
def probabilities_to_outcomes(prob):
    L=np.array(["H","D","A"]); return np.array([rng.choice(L,p=p) for p in prob])
def probabilities_to_odds(prob): return np.clip(MARGIN/prob,1.15,15.0)

def generate_fixtures(ids,outcomes):
    n=len(ids); home=rng.integers(100,200,n); away=rng.integers(200,300,n)
    return pd.DataFrame({
        "match_id":ids,"home_team_id":home,"away_team_id":away,
        "fixture_date_utc":make_calendar(n),"match_outcome":outcomes,
        "home_team_name":[f"Team_{i}" for i in home],"away_team_name":[f"Team_{i}" for i in away],
        "venue_name":[f"Stadium_{i%30+1}" for i in ids],"status":["FT"]*n
    })

def generate_odds(ids,prob):
    o=probabilities_to_odds(prob)
    df=pd.DataFrame({"match_id":ids,"odds_1_last":o[:,0],"odds_x_last":o[:,1],"odds_2_last":o[:,2]})
    for c in ("1","x","2"): df[f"odds_{c}_last_inv"]=1.0/df[f"odds_{c}_last"]
    return df

def generate_weather(ids):
    n=len(ids)
    return pd.DataFrame({
        "match_id":ids,"temperature_celsius":np.round(rng.normal(16,8,n),1),
        "humidity_percent":np.round(np.clip(rng.normal(62,18,n),5,98),1),
        "wind_speed_kph":np.round(np.clip(rng.normal(12,7,n),0,60),1),
        "conditions":rng.choice(["Clear","Cloudy","Rain","Windy"],size=n,p=[.3,.4,.2,.1])
    })

def generate_xg(ids,outcomes):
    n=len(ids); hx=np.zeros(n); ax=np.zeros(n)
    iH,iD,iA=(outcomes=="H"),(outcomes=="D"),(outcomes=="A")
    hx[iH]=rng.normal(1.8,.5,iH.sum()); ax[iH]=rng.normal(0.9,.4,iH.sum())
    hx[iD]=rng.normal(1.2,.4,iD.sum()); ax[iD]=rng.normal(1.1,.4,iD.sum())
    hx[iA]=rng.normal(0.9,.4,iA.sum()); ax[iA]=rng.normal(1.7,.5,iA.sum())
    hx=np.clip(hx,.05,None); ax=np.clip(ax,.05,None)
    def shots(x): return np.clip(np.round((x/0.10)+rng.normal(0,2,len(x))).astype(int),1,40)
    hs,as_=shots(hx),shots(ax)
    return pd.DataFrame({
        "match_id":ids,"home_xg":np.round(hx,2),"away_xg":np.round(ax,2),
        "home_shots":hs,"away_shots":as_,
        "home_shots_on_target":np.clip(np.round(hs*(0.2+hx/(hs+1))).astype(int),0,hs),
        "away_shots_on_target":np.clip(np.round(as_*(0.2+ax/(as_+1))).astype(int),0,as_)
    })

def generate_referees(ids):
    n=len(ids)
    return pd.DataFrame({
        "match_id":ids,"referee_id":rng.integers(5000,5050,n),
        "yellow_cards":np.round(np.clip(rng.normal(3.2,1.6,n),0,12)).astype(int),
        "red_cards":np.clip(rng.binomial(1,0.08,n)*rng.choice([0,1,2],size=n,p=[.6,.35,.05]),0,2).astype(int)
    })

def generate_squads(ids):
    n=len(ids); h=rng.normal(70,6,n)
    return pd.DataFrame({"match_id":ids,
        "home_overall_rating":np.round(np.clip(h,50,95),1),
        "away_overall_rating":np.round(np.clip(h-rng.normal(3,6,n),45,92),1)})

def main():
    ensure_dirs()
    ids=np.arange(1,N_TOTAL+1,dtype=int)
    prob=generate_probabilities(N_TOTAL)
    outcomes=probabilities_to_outcomes(prob)
    files={"fixtures":generate_fixtures(ids,outcomes),"odds":generate_odds(ids,prob),
           "weather":generate_weather(ids),"xg":generate_xg(ids,outcomes),
           "referees":generate_referees(ids),"squads":generate_squads(ids)}
    shapes={}
    for k,df in files.items():
        (RAW/f"{k}.csv").parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(RAW/f"{k}.csv",index=False); shapes[k]=len(df)
    print("[DONE] Mock dataset regenerated (OVERWRITE).")
    print("       rows:",shapes)
    print("       outcome dist:",pd.Series(outcomes).value_counts().to_dict())

if __name__=="__main__": main()
