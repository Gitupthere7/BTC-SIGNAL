#!/usr/bin/env python3
“””
BTC/USD Signal Engine — GitHub Actions Cloud Version
Runs once per trigger (GitHub handles the schedule).
Config comes from environment variables (GitHub Secrets).
State persisted via cache so it only alerts on signal changes.
“””

import os, json, time, requests, numpy as np
from datetime import datetime, timezone
from pathlib import Path

# ── Config from environment variables (set in GitHub Secrets) ──────────────

TELEGRAM_TOKEN  = os.environ.get(“TELEGRAM_TOKEN”, “”)
TELEGRAM_CHAT_ID= os.environ.get(“TELEGRAM_CHAT_ID”, “”)
ACCOUNT_CAPITAL = float(os.environ.get(“ACCOUNT_CAPITAL”, “10000”))
LEVERAGE        = int(os.environ.get(“LEVERAGE”, “5”))
MIN_CONFIDENCE  = int(os.environ.get(“MIN_CONFIDENCE”, “70”))
RISK_PCT        = float(os.environ.get(“RISK_PER_TRADE”, “2.0”))
TP_PCT          = float(os.environ.get(“TP_PCT”, “2.5”))
SL_PCT          = float(os.environ.get(“SL_PCT”, “2.0”))

# ── State file (persisted between runs via GitHub Actions cache) ────────────

STATE_DIR  = Path(”.state”)
STATE_FILE = STATE_DIR / “last_signal.json”

def load_state():
try:
return json.loads(STATE_FILE.read_text())
except Exception:
return {“signal”: None, “confidence”: 0, “alerted_at”: None}

def save_state(state):
STATE_DIR.mkdir(exist_ok=True)
STATE_FILE.write_text(json.dumps(state))

# ── Utilities ───────────────────────────────────────────────────────────────

def now_utc():
return datetime.now(timezone.utc).strftime(”%H:%M UTC”)

def log(msg):
ts = datetime.now(timezone.utc).strftime(”%H:%M:%S”)
print(f”[{ts}] {msg}”)

# ── Kraken data fetch ───────────────────────────────────────────────────────

def fetch_candles(pair=“XBTUSD”, interval=15, count=300):
since = int(time.time()) - interval * 60 * count
url = (f”https://api.kraken.com/0/public/OHLC”
f”?pair={pair}&interval={interval}&since={since}”)
r = requests.get(url, timeout=20)
r.raise_for_status()
data = r.json()
if data.get(“error”):
raise ValueError(f”Kraken error: {data[‘error’]}”)
key = next(k for k in data[“result”] if k != “last”)
return [
{“t”: int(c[0]), “o”: float(c[1]), “h”: float(c[2]),
“l”: float(c[3]), “c”: float(c[4]), “v”: float(c[6])}
for c in data[“result”][key]
]

# ── Indicators ───────────────────────────────────────────────────────────────

def ema(a, p):
a = np.array(a, float)
out = np.full(len(a), np.nan)
if p > len(a): return out
out[p-1] = np.mean(a[:p])
k = 2/(p+1)
for i in range(p, len(a)):
out[i] = a[i]*k + out[i-1]*(1-k)
return out

def rsi(c, p=14):
out = np.full(len(c), np.nan)
for i in range(p, len(c)):
d = np.diff(c[i-p:i+1])
g = d[d>0].sum(); l = -d[d<0].sum()
out[i] = 100 - 100/(1 + (g/l if l else 100))
return out

def macd_hist(c, f=12, s=26, sg=9):
ml = ema(c,f) - ema(c,s)
sl = np.full(len(c), np.nan)
buf, last, k = [], np.nan, 2/(sg+1)
for i, v in enumerate(ml):
if np.isnan(v): continue
buf.append(v)
if len(buf) < sg: continue
if len(buf) == sg: last = np.mean(buf)
else: last = v*k + last*(1-k)
sl[i] = last
return np.where(~np.isnan(ml) & ~np.isnan(sl), ml-sl, np.nan)

def bollinger(c, p=20, m=2.0):
c = np.array(c, float)
mid = np.full(len(c), np.nan)
up  = np.full(len(c), np.nan)
lo  = np.full(len(c), np.nan)
for i in range(p-1, len(c)):
w = c[i-p+1:i+1]; mn = w.mean(); sd = w.std()
mid[i]=mn; up[i]=mn+m*sd; lo[i]=mn-m*sd
return up, mid, lo

def stoch(cd, kp=14, dp=3):
hi=np.array([c[“h”] for c in cd]); lo=np.array([c[“l”] for c in cd])
cl=np.array([c[“c”] for c in cd]); k=np.full(len(cd), np.nan)
for i in range(kp-1, len(cd)):
hh=hi[i-kp+1:i+1].max(); ll=lo[i-kp+1:i+1].min()
k[i] = 50.0 if hh==ll else (cl[i]-ll)/(hh-ll)*100
d = np.full(len(cd), np.nan)
for i in range(dp-1, len(k)):
w=k[i-dp+1:i+1]
if not np.any(np.isnan(w)): d[i]=w.mean()
return k, d

def adx_calc(cd, p=14):
hi=np.array([c[“h”] for c in cd]); lo=np.array([c[“l”] for c in cd])
cl=np.array([c[“c”] for c in cd])
tr = np.maximum(hi-lo, np.maximum(abs(hi-np.roll(cl,1)), abs(lo-np.roll(cl,1))))
tr[0] = hi[0]-lo[0]
pl=np.zeros(len(cd)); mn=np.zeros(len(cd))
for i in range(1, len(cd)):
u=hi[i]-hi[i-1]; d=lo[i-1]-lo[i]
pl[i]=max(u,0) if u>d else 0
mn[i]=max(d,0) if d>u else 0
av = np.where(ema(tr,p)<0.001, 0.001, ema(tr,p))
pdi=ema(pl,p)/av*100; mdi=ema(mn,p)/av*100
dx = np.where((pdi+mdi)>0, abs(pdi-mdi)/(pdi+mdi)*100, 0.0)
return ema(dx,p), pdi, mdi

def obv_calc(cd):
cl=np.array([c[“c”] for c in cd]); vl=np.array([c[“v”] for c in cd])
o=np.zeros(len(cd))
for i in range(1, len(cd)):
o[i]=o[i-1]+(vl[i] if cl[i]>cl[i-1] else -vl[i] if cl[i]<cl[i-1] else 0)
return o

# ── Signal engine ────────────────────────────────────────────────────────────

def compute(candles):
if len(candles) < 220:
return None

```
c = np.array([x["c"] for x in candles])
n=len(c); i=n-1; pv=n-2; pv2=n-3; price=c[i]

e8=ema(c,8); e21=ema(c,21); e50=ema(c,50); e200=ema(c,200)
rv=rsi(c,14); mh=macd_hist(c)
adxv,pdi,mdi = adx_calc(candles,14)
_,bbm,_ = bollinger(c,20,2.0)
sk,sd = stoch(candles,14,3)
ov=obv_calc(candles); ove=ema(ov,21)

e8v=e8[i]; e8p=e8[pv]; e21v=e21[i]; e21p=e21[pv]
e50v=e50[i]; e200v=e200[i]; e200p=e200[pv]
rvv=rv[i]; rvp=rv[pv]; rvp2=rv[pv2]
mhv=mh[i]; mhvp=mh[pv]
adx_v=adxv[i]; pdi_v=pdi[i]; mdi_v=mdi[i]
skv=sk[i]; sdv=sd[i]
ovv=ov[i]; ovev=ove[i]
bbmv=bbm[i]

if any(np.isnan(x) for x in [e8v,e21v,e50v,e200v,rvv,mhv,adx_v]):
    return None

# Regime
regime_bull = e200v>e200p and price>e200v and e50v>e200v
regime_bear = e200v<e200p and price<e200v and e50v<e200v

# ── LONG scoring ─────────────────────────────────────────────────────
long_checks = [
    ("EMA ribbon stacked bull (8 > 21 > 50)",      e8v>e21v>e50v,                     20),
    ("ADX strong trend, +DI leading -DI",           adx_v>20 and pdi_v>mdi_v+3,        15),
    ("RSI dipped recently (was below 58)",          rvp<58 or rvp2<58,                  8),
    ("RSI recovering into 44–65 zone",              44<rvv<65 and rvv>rvp,              15),
    ("MACD histogram positive",                     mhv>0,                              10),
    ("MACD fresh bullish cross above zero",         mhv>0 and mhvp<=0,                 10),
    ("Price bouncing above EMA 21",                 price>e21v and price>c[pv],          8),
    ("OBV above trend — volume confirming",         not np.isnan(ovev) and ovv>ovev*0.97, 7),
    ("Stochastic bullish, not overbought",          not np.isnan(skv) and skv>sdv and skv<75, 5),
    ("Three consecutive rising candles",            c[i]>c[pv]>c[pv2],                  5),
]

# ── SHORT scoring ─────────────────────────────────────────────────────
short_checks = [
    ("EMA ribbon stacked bear (8 < 21 < 50)",      e8v<e21v<e50v,                     20),
    ("ADX strong trend, -DI leading +DI",           adx_v>20 and mdi_v>pdi_v+3,        15),
    ("RSI was elevated recently (was above 52)",    rvp>52 or rvp2>52,                  8),
    ("RSI falling into 35–56 zone",                 35<rvv<56 and rvv<rvp,              15),
    ("MACD histogram negative",                     mhv<0,                              10),
    ("MACD fresh bearish cross below zero",         mhv<0 and mhvp>=0,                 10),
    ("Price rejected below EMA 21",                 price<e21v and price<c[pv],          8),
    ("OBV below trend — selling pressure",          not np.isnan(ovev) and ovv<ovev*1.03, 7),
    ("Stochastic bearish, not oversold",            not np.isnan(skv) and skv<sdv and skv>25, 5),
    ("Three consecutive falling candles",           c[i]<c[pv]<c[pv2],                  5),
]

def score(checks):
    total = sum(w for _,_,w in checks)
    hit   = sum(w for _,v,w in checks if v)
    fired = [name for name,v,_ in checks if v]
    return round(hit/total*100), fired

lconf, lfired = score(long_checks)
sconf, sfired = score(short_checks)

long_valid  = (regime_bull and e8v>e21v>e50v and
               adx_v>20 and pdi_v>mdi_v and
               44<rvv<65 and mhv>0 and price>e21v)

short_valid = (regime_bear and e8v<e21v<e50v and
               adx_v>20 and mdi_v>pdi_v and
               35<rvv<56 and mhv<0 and price<e21v)

if long_valid and lconf>=MIN_CONFIDENCE and lconf>=sconf:
    signal, conf, reasons = "LONG",  lconf, lfired
elif short_valid and sconf>=MIN_CONFIDENCE and sconf>lconf:
    signal, conf, reasons = "SHORT", sconf, sfired
else:
    signal, conf, reasons = "STAY OUT", max(lconf,sconf), []

# Trade levels
tp = price*(1+TP_PCT/100) if signal=="LONG"  else price*(1-TP_PCT/100)
sl = price*(1-SL_PCT/100) if signal=="LONG"  else price*(1+SL_PCT/100)
if signal=="STAY OUT":
    tp=price*(1+TP_PCT/100); sl=price*(1-SL_PCT/100)

# Position sizing
risk_usd   = ACCOUNT_CAPITAL * (RISK_PCT/100)
notional   = risk_usd / (SL_PCT/100)
margin_req = notional / LEVERAGE
btc_qty    = notional / price
liq_dist   = (1/LEVERAGE)*price*0.90
liq_price  = (price-liq_dist) if signal=="LONG" else (price+liq_dist)

return {
    "signal":     signal,
    "confidence": conf,
    "price":      price,
    "tp":         tp,
    "sl":         sl,
    "reasons":    reasons,
    "risk_usd":   risk_usd,
    "notional":   notional,
    "margin_req": margin_req,
    "btc_qty":    btc_qty,
    "liq_price":  liq_price,
    "regime":     "UPTREND" if regime_bull else "DOWNTREND" if regime_bear else "RANGING",
    "adx":        round(adx_v,1),
    "rsi":        round(rvv,1),
    "macd":       round(mhv,2),
    "e200_slope": "Rising ↑" if e200v>e200p else "Falling ↓",
}
```

# ── Telegram ─────────────────────────────────────────────────────────────────

def send(message):
if not TELEGRAM_TOKEN:
print(“TELEGRAM NOT SET — message would be:\n” + message)
return
requests.post(
f”https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage”,
json={“chat_id”: TELEGRAM_CHAT_ID, “text”: message, “parse_mode”: “HTML”},
timeout=10
).raise_for_status()

def build_alert(sig, prev):
p=sig[“price”]; tp=sig[“tp”]; sl_p=sig[“sl”]
s=sig[“signal”]

```
if s == "STAY OUT":
    return (
        f"⚪ <b>SIGNAL CLEARED — BTC/USD</b>\n"
        f"{'─'*28}\n"
        f"Was: {prev}  →  Now: STAY OUT\n"
        f"Price: ${p:,.2f}\n"
        f"Regime: {sig['regime']}  |  ADX: {sig['adx']}  |  RSI: {sig['rsi']}\n"
        f"Confidence below {MIN_CONFIDENCE}% — no trade.\n\n"
        f"⏱ {now_utc()}"
    )

emoji = "🟢" if s=="LONG" else "🔴"
direction = "▲ LONG" if s=="LONG" else "▼ SHORT"
tp_txt = f"+{TP_PCT}%" if s=="LONG" else f"-{TP_PCT}%"
sl_txt = f"-{SL_PCT}%" if s=="LONG" else f"+{SL_PCT}%"
action = ("BUY — open LONG position\non Kraken Pro margin"
          if s=="LONG" else
          "SELL — open SHORT position\non Kraken Pro margin")
reasons = "\n".join(f"  ▸ {r}" for r in sig["reasons"])

return (
    f"{emoji} <b>{direction}  —  BTC/USD</b>\n"
    f"{'─'*30}\n"
    f"Price      ${p:>12,.2f}\n"
    f"Target     ${tp:>12,.2f}   ({tp_txt})\n"
    f"Stop       ${sl_p:>12,.2f}   ({sl_txt})\n"
    f"Strength   {sig['confidence']}%"
    f"  |  ADX {sig['adx']}  |  RSI {sig['rsi']}\n"
    f"Regime     {sig['regime']}  —  EMA200 {sig['e200_slope']}\n\n"
    f"<b>Why {s}:</b>\n{reasons}\n\n"
    f"<b>Position ({LEVERAGE}x margin):</b>\n"
    f"  Notional     ${sig['notional']:>10,.2f}\n"
    f"  Margin req   ${sig['margin_req']:>10,.2f}\n"
    f"  BTC qty       {sig['btc_qty']:>10.6f}\n"
    f"  Max loss     ${sig['risk_usd']:>10,.2f}  (2% of capital)\n"
    f"  Est. liq.    ${sig['liq_price']:>10,.2f}  ⚠\n\n"
    f"<b>ACTION:  {action}</b>\n\n"
    f"⏱ {now_utc()}\n"
    f"<i>Signal only — verify on Kraken before executing.</i>"
)
```

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
log(“BTC Signal Engine — starting scan”)

```
state = load_state()
prev_signal = state.get("signal")
log(f"Previous signal: {prev_signal or 'None'}")

log("Fetching live data from Kraken...")
candles = fetch_candles()
log(f"✓ {len(candles)} candles. Latest price: ${candles[-1]['c']:,.2f}")

sig = compute(candles)
if sig is None:
    log("Indicators warming up — skipping this scan.")
    return

log(f"Signal: {sig['signal']}  Confidence: {sig['confidence']}%  "
    f"Regime: {sig['regime']}  RSI: {sig['rsi']}  ADX: {sig['adx']}")

should_alert = False

if sig["signal"] in ("LONG", "SHORT"):
    if sig["confidence"] >= MIN_CONFIDENCE:
        if sig["signal"] != prev_signal:
            should_alert = True
            log(f"→ Direction changed ({prev_signal} → {sig['signal']}). Alerting.")
        else:
            log(f"  Same direction as last alert ({prev_signal}). No repeat alert.")
    else:
        log(f"  Below {MIN_CONFIDENCE}% threshold. No alert.")

elif sig["signal"] == "STAY OUT" and prev_signal in ("LONG","SHORT"):
    should_alert = True
    log(f"→ Signal cleared. Notifying.")

if should_alert:
    msg = build_alert(sig, prev_signal)
    send(msg)
    log("✓ Telegram alert sent.")

save_state({"signal": sig["signal"], "confidence": sig["confidence"],
            "alerted_at": now_utc()})
log("Scan complete.")
```

if **name** == “**main**”:
main()
