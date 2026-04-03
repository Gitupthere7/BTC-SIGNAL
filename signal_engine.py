import os, json, time, math, requests, numpy as np
from datetime import datetime, timezone
from pathlib import Path

TELEGRAM_TOKEN   = os.environ.get(‘TELEGRAM_TOKEN’, ‘’)
TELEGRAM_CHAT_ID = os.environ.get(‘TELEGRAM_CHAT_ID’, ‘’)
ACCOUNT_CAPITAL  = float(os.environ.get(‘ACCOUNT_CAPITAL’, ‘100’))
LEVERAGE         = int(os.environ.get(‘LEVERAGE’, ‘10’))
MIN_CONFIDENCE   = int(os.environ.get(‘MIN_CONFIDENCE’, ‘4’))
RISK_PCT         = float(os.environ.get(‘RISK_PER_TRADE’, ‘2.0’))
TP_PCT           = float(os.environ.get(‘TP_PCT’, ‘2.0’))
SL_PCT           = float(os.environ.get(‘SL_PCT’, ‘1.0’))
ADX_MIN          = float(os.environ.get(‘ADX_MIN’, ‘18’))

STATE_DIR  = Path(’.state’)
STATE_FILE = STATE_DIR / ‘last_signal.json’

def load_state():
try:
return json.loads(STATE_FILE.read_text())
except Exception:
return {‘signal’: None, ‘entry’: None, ‘time’: None}

def save_state(data):
STATE_DIR.mkdir(exist_ok=True)
STATE_FILE.write_text(json.dumps(data))

def utc_now():
return datetime.now(timezone.utc).strftime(’%H:%M UTC’)

def log(msg):
ts = datetime.now(timezone.utc).strftime(’%H:%M:%S’)
print(’[’ + ts + ’] ’ + str(msg))

def fetch_candles(interval=5, count=500):
since = int(time.time()) - interval * 60 * count
url = (‘https://api.kraken.com/0/public/OHLC’
+ ‘?pair=XBTUSD&interval=’ + str(interval)
+ ‘&since=’ + str(since))
r = requests.get(url, timeout=20)
r.raise_for_status()
data = r.json()
if data.get(‘error’):
raise ValueError(’Kraken error: ’ + str(data[‘error’]))
key = next(k for k in data[‘result’] if k != ‘last’)
return [
{‘t’: int(c[0]), ‘h’: float(c[2]), ‘l’: float(c[3]),
‘c’: float(c[4]), ‘v’: float(c[6])}
for c in data[‘result’][key]
]

def ema(arr, p):
a = np.array(arr, float)
out = np.full(len(a), np.nan)
if p > len(a):
return out
out[p - 1] = np.mean(a[:p])
k = 2.0 / (p + 1)
for i in range(p, len(a)):
out[i] = a[i] * k + out[i - 1] * (1 - k)
return out

def rsi(closes, p=14):
out = np.full(len(closes), np.nan)
for i in range(p, len(closes)):
d = np.diff(closes[i - p: i + 1])
g = d[d > 0].sum()
l = -d[d < 0].sum()
out[i] = 100 - 100 / (1 + (g / l if l else 100))
return out

def macd_hist(closes):
ef = ema(closes, 12)
es = ema(closes, 26)
ml = np.where(~np.isnan(ef) & ~np.isnan(es), ef - es, np.nan)
sl = np.full(len(closes), np.nan)
buf = []
last = np.nan
k = 2.0 / 10
for i, v in enumerate(ml):
if np.isnan(v):
continue
buf.append(v)
if len(buf) < 9:
continue
if len(buf) == 9:
last = np.mean(buf)
else:
last = v * k + last * (1 - k)
sl[i] = last
return np.where(~np.isnan(ml) & ~np.isnan(sl), ml - sl, np.nan)

def adx_pdi_mdi(candles, p=14):
hi = np.array([c[‘h’] for c in candles])
lo = np.array([c[‘l’] for c in candles])
cl = np.array([c[‘c’] for c in candles])
n  = len(candles)
tr = np.zeros(n)
pl = np.zeros(n)
mn = np.zeros(n)
tr[0] = hi[0] - lo[0]
for i in range(1, n):
tr[i] = max(hi[i] - lo[i],
abs(hi[i] - cl[i - 1]),
abs(lo[i] - cl[i - 1]))
u = hi[i] - hi[i - 1]
d = lo[i - 1] - lo[i]
pl[i] = max(u, 0) if u > d else 0
mn[i] = max(d, 0) if d > u else 0
av  = np.where(ema(tr, p) < 0.001, 0.001, ema(tr, p))
pdi = ema(pl, p) / av * 100
mdi = ema(mn, p) / av * 100
dx  = np.where((pdi + mdi) > 0,
np.abs(pdi - mdi) / (pdi + mdi) * 100, 0.0)
return ema(dx, p), pdi, mdi

def vwap_daily(candles, bars_per_day=288):
n   = len(candles)
tp  = np.array([(c[‘h’] + c[‘l’] + c[‘c’]) / 3 for c in candles])
vol = np.array([c[‘v’] for c in candles])
out = np.full(n, np.nan)
for ds in range(0, n, bars_per_day):
de  = min(ds + bars_per_day, n)
ctv = np.cumsum(tp[ds:de] * vol[ds:de])
cv  = np.cumsum(vol[ds:de])
out[ds:de] = np.where(cv > 0, ctv / cv, tp[ds:de])
return out

def h1_ema(candles, period, bars_per_hour=12):
n   = len(candles)
cl  = np.array([c[‘c’] for c in candles])
h1c = []
h1i = []
for i in range(0, n, bars_per_hour):
end = min(i + bars_per_hour, n)
h1c.append(cl[end - 1])
h1i.append(end - 1)
h1v = ema(np.array(h1c), period)
out = np.full(n, np.nan)
for k, idx in enumerate(h1i):
prev = h1i[k - 1] + 1 if k > 0 else 0
out[prev: idx + 1] = h1v[k]
return out

def in_session(unix_ts):
hour   = datetime.fromtimestamp(unix_ts, tz=timezone.utc).hour
london = 8 <= hour < 16
ny     = 13 <= hour < 21
asia   = 0 <= hour < 4
return london or ny or asia

def compute(candles):
if len(candles) < 300:
return None
cl  = np.array([c[‘c’] for c in candles])
vl  = np.array([c[‘v’] for c in candles])
n   = len(cl)
i   = n - 1
pv  = n - 2
pv2 = n - 3

```
e9    = ema(cl, 9)
e21   = ema(cl, 21)
rv    = rsi(cl, 14)
mh    = macd_hist(cl)
vw    = vwap_daily(candles, 288)
vm    = ema(vl, 20)
adxv, pdi, mdi = adx_pdi_mdi(candles, 14)
h1e21 = h1_ema(candles, 21)
h1e50 = h1_ema(candles, 50)

price = cl[i]
needed = [e9[i], e21[i], rv[i], mh[i], vw[i], adxv[i]]
if any(np.isnan(x) for x in needed):
    return None

session_active = in_session(candles[i]['t'])
adx_ok         = adxv[i] >= ADX_MIN
trend_ok       = pdi[i] > mdi[i]

h1_bull = True
if not np.isnan(h1e21[i]) and not np.isnan(h1e50[i]):
    h1_bull = h1e21[i] > h1e50[i]

above_vwap   = price > vw[i]
rsi_pull     = ((rv[pv] < 52 or (pv2 >= 0 and rv[pv2] < 52))
                and rv[i] > 44 and rv[i] > rv[pv])
ema_cross    = e9[i] > e21[i] and e9[pv] <= e21[pv]
macd_pos     = mh[i] > 0
vol_ok       = (vl[i] > vm[i] * 0.8) if not np.isnan(vm[i]) else True
vwap_bounce  = ((price > vw[i] and cl[pv] < vw[pv])
                or (price > vw[i] and price > e9[i]))

score = sum([above_vwap, rsi_pull, ema_cross,
             macd_pos, vol_ok, vwap_bounce])

fired = []
if above_vwap:  fired.append('Price above VWAP')
if rsi_pull:    fired.append('RSI pulled back then recovering (' + str(round(rv[i], 1)) + ')')
if ema_cross:   fired.append('EMA9 just crossed above EMA21')
if macd_pos:    fired.append('MACD histogram positive (' + str(round(float(mh[i]), 2)) + ')')
if vol_ok:      fired.append('Volume above average')
if vwap_bounce: fired.append('Price bouncing off VWAP')
if adx_ok:      fired.append('ADX ' + str(round(adxv[i], 1)) + ' confirms trend')
if trend_ok:    fired.append('+DI > -DI buyers in control')
if h1_bull:     fired.append('1H trend bullish EMA21 above EMA50')

valid_long = (score >= MIN_CONFIDENCE
              and above_vwap
              and adx_ok
              and trend_ok
              and session_active)

vwap_break = (price < vw[i] and cl[pv] > vw[pv] and mh[i] < 0)

tp = price * (1 + TP_PCT / 100)
sl = price * (1 - SL_PCT / 100)

risk_usd   = ACCOUNT_CAPITAL * (RISK_PCT / 100)
notional   = risk_usd / (SL_PCT / 100)
margin_req = notional / LEVERAGE
btc_qty    = notional / price
liq_price  = price - (1 / LEVERAGE) * price * 0.90

return {
    'signal':     'LONG' if valid_long else 'WAIT',
    'vwap_break': vwap_break,
    'score':      score,
    'price':      price,
    'tp':         tp,
    'sl':         sl,
    'fired':      fired,
    'risk_usd':   risk_usd,
    'notional':   notional,
    'margin_req': margin_req,
    'btc_qty':    btc_qty,
    'liq_price':  liq_price,
    'adx':        round(adxv[i], 1),
    'pdi':        round(pdi[i], 1),
    'mdi':        round(mdi[i], 1),
    'rsi':        round(rv[i], 1),
    'macd':       round(float(mh[i]), 2),
    'vwap':       round(vw[i], 2),
    'session':    session_active,
    'adx_ok':     adx_ok,
    'h1_bull':    h1_bull,
}
```

def send(message):
if not TELEGRAM_TOKEN:
print(‘NO TOKEN - message:’)
print(message)
return
url = ‘https://api.telegram.org/bot’ + TELEGRAM_TOKEN + ‘/sendMessage’
requests.post(url,
json={‘chat_id’: TELEGRAM_CHAT_ID,
‘text’: message,
‘parse_mode’: ‘HTML’},
timeout=10).raise_for_status()

def build_long_alert(sig):
p  = sig[‘price’]
tp = sig[‘tp’]
sl = sig[‘sl’]
reasons = ‘\n’.join(’  - ’ + r for r in sig[‘fired’])
lines = [
‘LONG - BTC/USD’,
‘=’ * 28,
‘Price    $’ + ‘{:,.2f}’.format(p),
‘Target   $’ + ‘{:,.2f}’.format(tp) + ’  (+’ + str(TP_PCT) + ‘%)’,
‘Stop     $’ + ‘{:,.2f}’.format(sl) + ’  (-’ + str(SL_PCT) + ‘%)’,
‘VWAP     $’ + ‘{:,.2f}’.format(sig[‘vwap’]),
‘ADX ’ + str(sig[‘adx’]) + ’  RSI ’ + str(sig[‘rsi’]),
‘’,
‘Why:’,
reasons,
‘’,
‘Position (’ + str(LEVERAGE) + ‘x margin):’,
’  Notional  $’ + ‘{:,.2f}’.format(sig[‘notional’]),
’  Margin    $’ + ‘{:,.2f}’.format(sig[‘margin_req’]),
’  BTC qty    ’ + ‘{:.6f}’.format(sig[‘btc_qty’]),
’  Max loss  $’ + ‘{:,.2f}’.format(sig[‘risk_usd’]),
’  Est liq   $’ + ‘{:,.2f}’.format(sig[‘liq_price’]),
‘’,
‘ACTION: BUY on Kraken margin’,
‘TP +’ + str(TP_PCT) + ‘%  |  SL -’ + str(SL_PCT) + ‘%’,
‘’,
utc_now(),
‘Signal only. Verify before trading.’,
]
return ‘\n’.join(lines)

def build_exit_alert(sig, entry_price):
p   = sig[‘price’]
pnl = (p - entry_price) / entry_price * 100
lines = [
‘EXIT SIGNAL - BTC/USD’,
‘=’ * 28,
‘Price    $’ + ‘{:,.2f}’.format(p),
‘Entry    $’ + ‘{:,.2f}’.format(entry_price),
‘P&L      ’ + ‘{:+.2f}’.format(pnl) + ‘%’,
‘’,
‘Reason: Price broke below VWAP’,
’        with MACD turning negative.’,
‘’,
‘Consider closing if not at TP/SL.’,
‘’,
utc_now(),
]
return ‘\n’.join(lines)

def build_cleared_alert(sig):
missing = []
if not sig[‘adx_ok’]:
missing.append(‘ADX ’ + str(sig[‘adx’]) + ’ below ’ + str(int(ADX_MIN)) + ’ - market ranging’)
if sig[‘adx_ok’] and sig[‘pdi’] <= sig[‘mdi’]:
missing.append(’+DI < -DI - sellers in control’)
if not sig[‘h1_bull’]:
missing.append(‘1H trend not bullish’)
if not sig[‘session’]:
missing.append(‘Outside active trading session’)
if sig[‘score’] < MIN_CONFIDENCE:
missing.append(‘Score ’ + str(sig[‘score’]) + ’ of ’ + str(MIN_CONFIDENCE) + ’ needed’)
if not missing:
missing.append(‘Conditions no longer met’)
miss_str = ‘\n’.join(’  - ’ + m for m in missing)
lines = [
‘SIGNAL CLEARED - BTC/USD’,
‘=’ * 28,
‘Price    $’ + ‘{:,.2f}’.format(sig[‘price’]),
‘Was: LONG  Now: WAIT’,
‘’,
‘Not meeting criteria:’,
miss_str,
‘’,
utc_now(),
]
return ‘\n’.join(lines)

def main():
log(‘BTC 5-min VWAP Scalper starting’)
state = load_state()
prev  = state.get(‘signal’)
entry = state.get(‘entry’)
log(’Previous signal: ’ + str(prev))

```
log('Fetching 5-min candles from Kraken...')
candles = fetch_candles(interval=5, count=500)
log('Loaded ' + str(len(candles)) + ' candles. Price: $'
    + '{:,.2f}'.format(candles[-1]['c']))

sig = compute(candles)
if sig is None:
    log('Indicators warming up - skipping')
    return

log('Signal: ' + sig['signal']
    + '  Score: ' + str(sig['score']) + '/6'
    + '  ADX: ' + str(sig['adx'])
    + '  RSI: ' + str(sig['rsi'])
    + '  Session: ' + str(sig['session']))

if sig['signal'] == 'LONG' and prev != 'LONG':
    send(build_long_alert(sig))
    log('LONG alert sent')
    save_state({'signal': 'LONG', 'entry': sig['price'], 'time': utc_now()})

elif sig['signal'] == 'LONG' and prev == 'LONG':
    log('Still LONG - holding')
    if sig['vwap_break'] and entry:
        send(build_exit_alert(sig, entry))
        log('Early exit alert sent - VWAP break')
        save_state({'signal': 'WAIT', 'entry': None, 'time': utc_now()})

elif sig['signal'] == 'WAIT' and prev == 'LONG':
    send(build_cleared_alert(sig))
    log('Signal cleared - notified')
    save_state({'signal': 'WAIT', 'entry': None, 'time': utc_now()})

else:
    log('WAIT - no change')

log('Scan complete')
```

main()
