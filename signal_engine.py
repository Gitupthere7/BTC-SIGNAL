import os, json, time, requests, numpy as np
from datetime import datetime, timezone
from pathlib import Path

TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
ACCOUNT_CAPITAL  = float(os.environ.get('ACCOUNT_CAPITAL', '100'))
LEVERAGE         = int(os.environ.get('LEVERAGE', '10'))
MIN_CONFIDENCE   = int(os.environ.get('MIN_CONFIDENCE', '4'))
RISK_PCT         = float(os.environ.get('RISK_PER_TRADE', '2.0'))
TP_PCT           = float(os.environ.get('TP_PCT', '0.2'))
SL_PCT           = float(os.environ.get('SL_PCT', '0.1'))
ADX_MIN          = float(os.environ.get('ADX_MIN', '18'))

STATE_DIR  = Path('.state')
STATE_FILE = STATE_DIR / 'last_signal.json'

def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {'signal': None, 'entry': None, 'tp': None,
                'sl': None, 'time': None, 'consec_l': 0, 'consec_s': 0}

def save_state(data):
    STATE_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(data))

def utc_now():
    return datetime.now(timezone.utc).strftime('%H:%M UTC')

def log(msg):
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    print('[' + ts + '] ' + str(msg))

def fetch_candles(interval=5, count=500):
    since = int(time.time()) - interval * 60 * count
    url = ('https://api.kraken.com/0/public/OHLC'
           + '?pair=XBTUSD&interval=' + str(interval)
           + '&since=' + str(since))
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get('error'):
        raise ValueError('Kraken error: ' + str(data['error']))
    key = next(k for k in data['result'] if k != 'last')
    return [
        {'t': int(c[0]), 'h': float(c[2]), 'l': float(c[3]),
         'c': float(c[4]), 'v': float(c[6])}
        for c in data['result'][key]
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
    hi = np.array([c['h'] for c in candles])
    lo = np.array([c['l'] for c in candles])
    cl = np.array([c['c'] for c in candles])
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
    tp  = np.array([(c['h'] + c['l'] + c['c']) / 3 for c in candles])
    vol = np.array([c['v'] for c in candles])
    out = np.full(n, np.nan)
    for ds in range(0, n, bars_per_day):
        de  = min(ds + bars_per_day, n)
        ctv = np.cumsum(tp[ds:de] * vol[ds:de])
        cv  = np.cumsum(vol[ds:de])
        out[ds:de] = np.where(cv > 0, ctv / cv, tp[ds:de])
    return out

def h1_ema(candles, period, bars_per_hour=12):
    n   = len(candles)
    cl  = np.array([c['c'] for c in candles])
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
    cl  = np.array([c['c'] for c in candles])
    vl  = np.array([c['v'] for c in candles])
    n   = len(cl)
    i   = n - 1
    pv  = n - 2
    pv2 = n - 3

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
    if any(np.isnan(x) for x in [e9[i], e21[i], rv[i], mh[i], vw[i], adxv[i]]):
        return None

    session_active = in_session(candles[i]['t'])
    adx_ok         = adxv[i] >= ADX_MIN
    pdi_wins       = pdi[i] > mdi[i]
    mdi_wins       = mdi[i] > pdi[i]
    h1_bull = True
    h1_bear = True
    if not np.isnan(h1e21[i]) and not np.isnan(h1e50[i]):
        h1_bull = h1e21[i] > h1e50[i]
        h1_bear = h1e21[i] < h1e50[i]

    # LONG conditions
    above_vwap  = price > vw[i]
    rsi_pull    = ((rv[pv] < 52 or (pv2 >= 0 and rv[pv2] < 52))
                   and rv[i] > 44 and rv[i] > rv[pv])
    ema_cross_l = e9[i] > e21[i] and e9[pv] <= e21[pv]
    macd_pos    = mh[i] > 0
    vol_ok      = (vl[i] > vm[i] * 0.8) if not np.isnan(vm[i]) else True
    vwap_bou_l  = ((price > vw[i] and cl[pv] < vw[pv])
                   or (price > vw[i] and price > e9[i]))
    long_score  = sum([above_vwap, rsi_pull, ema_cross_l,
                       macd_pos, vol_ok, vwap_bou_l])

    # SHORT conditions (mirror)
    below_vwap  = price < vw[i]
    rsi_fall    = ((rv[pv] > 48 or (pv2 >= 0 and rv[pv2] > 48))
                   and rv[i] < 56 and rv[i] < rv[pv])
    ema_cross_s = e9[i] < e21[i] and e9[pv] >= e21[pv]
    macd_neg    = mh[i] < 0
    vwap_bou_s  = ((price < vw[i] and cl[pv] > vw[pv])
                   or (price < vw[i] and price < e9[i]))
    short_score = sum([below_vwap, rsi_fall, ema_cross_s,
                       macd_neg, vol_ok, vwap_bou_s])

    long_valid  = (long_score >= MIN_CONFIDENCE and above_vwap
                   and adx_ok and pdi_wins and h1_bull and session_active)
    short_valid = (short_score >= MIN_CONFIDENCE and below_vwap
                   and adx_ok and mdi_wins and h1_bear and session_active)

    long_fired = []
    if above_vwap:  long_fired.append('Price above VWAP')
    if rsi_pull:    long_fired.append('RSI pulled back then recovering (' + str(round(rv[i], 1)) + ')')
    if ema_cross_l: long_fired.append('EMA9 crossed above EMA21')
    if macd_pos:    long_fired.append('MACD positive (' + str(round(float(mh[i]), 2)) + ')')
    if vol_ok:      long_fired.append('Volume above average')
    if vwap_bou_l:  long_fired.append('Price bouncing above VWAP')
    if adx_ok:      long_fired.append('ADX ' + str(round(adxv[i], 1)) + ' trend confirmed')
    if pdi_wins:    long_fired.append('+DI > -DI buyers in control')
    if h1_bull:     long_fired.append('1H EMA21 above EMA50 bullish')

    short_fired = []
    if below_vwap:  short_fired.append('Price below VWAP')
    if rsi_fall:    short_fired.append('RSI was elevated now falling (' + str(round(rv[i], 1)) + ')')
    if ema_cross_s: short_fired.append('EMA9 crossed below EMA21')
    if macd_neg:    short_fired.append('MACD negative (' + str(round(float(mh[i]), 2)) + ')')
    if vol_ok:      short_fired.append('Volume above average')
    if vwap_bou_s:  short_fired.append('Price rejected below VWAP')
    if adx_ok:      short_fired.append('ADX ' + str(round(adxv[i], 1)) + ' trend confirmed')
    if mdi_wins:    short_fired.append('-DI > +DI sellers in control')
    if h1_bear:     short_fired.append('1H EMA21 below EMA50 bearish')

    tp_l = price * (1 + TP_PCT / 100)
    sl_l = price * (1 - SL_PCT / 100)
    tp_s = price * (1 - TP_PCT / 100)
    sl_s = price * (1 + SL_PCT / 100)

    risk_usd   = ACCOUNT_CAPITAL * (RISK_PCT / 100)
    notional   = risk_usd / (SL_PCT / 100)
    margin_req = notional / LEVERAGE
    btc_qty    = notional / price
    liq_l      = price - (1 / LEVERAGE) * price * 0.90
    liq_s      = price + (1 / LEVERAGE) * price * 0.90

    signal = 'WAIT'
    if long_valid:
        signal = 'LONG'
    elif short_valid:
        signal = 'SHORT'

    vwap_break_l = (price < vw[i] and cl[pv] > vw[pv] and mh[i] < 0)
    vwap_break_s = (price > vw[i] and cl[pv] < vw[pv] and mh[i] > 0)

    return {
        'signal':      signal,
        'long_valid':  long_valid,
        'short_valid': short_valid,
        'long_score':  long_score,
        'short_score': short_score,
        'long_fired':  long_fired,
        'short_fired': short_fired,
        'price':       price,
        'tp_l':        tp_l,
        'sl_l':        sl_l,
        'tp_s':        tp_s,
        'sl_s':        sl_s,
        'vwap_break_l': vwap_break_l,
        'vwap_break_s': vwap_break_s,
        'risk_usd':    risk_usd,
        'notional':    notional,
        'margin_req':  margin_req,
        'btc_qty':     btc_qty,
        'liq_l':       liq_l,
        'liq_s':       liq_s,
        'adx':         round(adxv[i], 1),
        'pdi':         round(pdi[i], 1),
        'mdi':         round(mdi[i], 1),
        'rsi':         round(rv[i], 1),
        'macd':        round(float(mh[i]), 2),
        'vwap':        round(vw[i], 2),
        'session':     session_active,
        'adx_ok':      adx_ok,
        'h1_bull':     h1_bull,
        'h1_bear':     h1_bear,
    }

def send(message):
    if not TELEGRAM_TOKEN:
        print('NO TOKEN - message:')
        print(message)
        return
    url = 'https://api.telegram.org/bot' + TELEGRAM_TOKEN + '/sendMessage'
    requests.post(url,
        json={'chat_id': TELEGRAM_CHAT_ID,
              'text': message,
              'parse_mode': 'HTML'},
        timeout=10).raise_for_status()

def build_long_alert(sig):
    reasons = '\n'.join('  - ' + r for r in sig['long_fired'])
    p = sig['price']
    lines = [
        'LONG - BTC/USD',
        '=' * 28,
        'Price    $' + '{:,.2f}'.format(p),
        'Target   $' + '{:,.2f}'.format(sig['tp_l']) + '  (+' + str(TP_PCT) + '%)',
        'Stop     $' + '{:,.2f}'.format(sig['sl_l']) + '  (-' + str(SL_PCT) + '%)',
        'VWAP     $' + '{:,.2f}'.format(sig['vwap']),
        'ADX ' + str(sig['adx']) + '  RSI ' + str(sig['rsi']),
        '',
        'Why LONG:',
        reasons,
        '',
        'Position (' + str(LEVERAGE) + 'x margin):',
        '  Notional  $' + '{:,.2f}'.format(sig['notional']),
        '  Margin    $' + '{:,.2f}'.format(sig['margin_req']),
        '  BTC qty    ' + '{:.6f}'.format(sig['btc_qty']),
        '  Max loss  $' + '{:,.2f}'.format(sig['risk_usd']),
        '  Est liq   $' + '{:,.2f}'.format(sig['liq_l']),
        '',
        'ACTION: BUY on Kraken margin',
        'TP +' + str(TP_PCT) + '%  |  SL -' + str(SL_PCT) + '%',
        '',
        utc_now(),
        'Signal only. Verify before trading.',
    ]
    return '\n'.join(lines)

def build_short_alert(sig):
    reasons = '\n'.join('  - ' + r for r in sig['short_fired'])
    p = sig['price']
    lines = [
        'SHORT - BTC/USD',
        '=' * 28,
        'Price    $' + '{:,.2f}'.format(p),
        'Target   $' + '{:,.2f}'.format(sig['tp_s']) + '  (-' + str(TP_PCT) + '%)',
        'Stop     $' + '{:,.2f}'.format(sig['sl_s']) + '  (+' + str(SL_PCT) + '%)',
        'VWAP     $' + '{:,.2f}'.format(sig['vwap']),
        'ADX ' + str(sig['adx']) + '  RSI ' + str(sig['rsi']),
        '',
        'Why SHORT:',
        reasons,
        '',
        'Position (' + str(LEVERAGE) + 'x margin):',
        '  Notional  $' + '{:,.2f}'.format(sig['notional']),
        '  Margin    $' + '{:,.2f}'.format(sig['margin_req']),
        '  BTC qty    ' + '{:.6f}'.format(sig['btc_qty']),
        '  Max loss  $' + '{:,.2f}'.format(sig['risk_usd']),
        '  Est liq   $' + '{:,.2f}'.format(sig['liq_s']),
        '',
        'ACTION: SELL on Kraken margin',
        'TP -' + str(TP_PCT) + '%  |  SL +' + str(SL_PCT) + '%',
        '',
        utc_now(),
        'Signal only. Verify before trading.',
    ]
    return '\n'.join(lines)

def build_tp_alert(direction, entry, exit_p):
    pnl = abs(exit_p - entry) / entry * 100
    lines = [
        'TAKE PROFIT HIT - BTC/USD',
        '=' * 28,
        'Direction  ' + direction,
        'Entry    $' + '{:,.2f}'.format(entry),
        'Exit     $' + '{:,.2f}'.format(exit_p),
        'P&L      +' + '{:.2f}'.format(pnl) + '%',
        '',
        'Close your position now.',
        '',
        utc_now(),
    ]
    return '\n'.join(lines)

def build_sl_alert(direction, entry, exit_p):
    pnl = abs(exit_p - entry) / entry * 100
    lines = [
        'STOP LOSS HIT - BTC/USD',
        '=' * 28,
        'Direction  ' + direction,
        'Entry    $' + '{:,.2f}'.format(entry),
        'Exit     $' + '{:,.2f}'.format(exit_p),
        'P&L      -' + '{:.2f}'.format(pnl) + '%',
        '',
        'Close your position now.',
        '',
        utc_now(),
    ]
    return '\n'.join(lines)

def build_vwap_exit_alert(direction, entry, price):
    pnl = (price - entry) / entry * 100 if direction == 'LONG' else (entry - price) / entry * 100
    lines = [
        'EARLY EXIT - BTC/USD',
        '=' * 28,
        'Direction  ' + direction,
        'Entry    $' + '{:,.2f}'.format(entry),
        'Price    $' + '{:,.2f}'.format(price),
        'P&L now  ' + '{:+.2f}'.format(pnl) + '%',
        '',
        'VWAP flipped against your trade.',
        'Consider closing early.',
        '',
        utc_now(),
    ]
    return '\n'.join(lines)

def build_cleared_alert(sig, prev):
    lines = [
        'SIGNAL CLEARED - BTC/USD',
        '=' * 28,
        'Price    $' + '{:,.2f}'.format(sig['price']),
        'Was: ' + prev + '  Now: WAIT',
        '',
        'ADX ' + str(sig['adx']) + '  RSI ' + str(sig['rsi']),
        '',
        utc_now(),
    ]
    return '\n'.join(lines)

def reset_state():
    save_state({'signal': None, 'entry': None, 'tp': None,
                'sl': None, 'time': None, 'consec_l': 0, 'consec_s': 0})

def main():
    log('BTC VWAP Scalper - Long + Short')
    state    = load_state()
    prev     = state.get('signal')
    entry    = state.get('entry')
    tp_lvl   = state.get('tp')
    sl_lvl   = state.get('sl')
    consec_l = state.get('consec_l', 0)
    consec_s = state.get('consec_s', 0)
    log('Prev: ' + str(prev) + '  ConsecL: ' + str(consec_l) + '  ConsecS: ' + str(consec_s))

    log('Fetching candles...')
    candles = fetch_candles(interval=5, count=500)
    price   = candles[-1]['c']
    log('Loaded ' + str(len(candles)) + ' candles. Price: $' + '{:,.2f}'.format(price))

    sig = compute(candles)
    if sig is None:
        log('Warming up - skipping')
        return

    log('Signal: ' + sig['signal']
        + '  LScore: ' + str(sig['long_score'])
        + '  SScore: ' + str(sig['short_score'])
        + '  ADX: ' + str(sig['adx'])
        + '  PDI: ' + str(sig['pdi'])
        + '  MDI: ' + str(sig['mdi'])
        + '  RSI: ' + str(sig['rsi'])
        + '  H1Bear: ' + str(sig['h1_bear'])
        + '  Session: ' + str(sig['session']))


    # IN A LONG TRADE - monitor for exit
    if prev == 'LONG' and entry and tp_lvl and sl_lvl:
        if price >= tp_lvl:
            send(build_tp_alert('LONG', entry, price))
            log('TP hit - LONG closed')
            reset_state()
            return
        if price <= sl_lvl:
            send(build_sl_alert('LONG', entry, price))
            log('SL hit - LONG closed')
            reset_state()
            return
        if sig['vwap_break_l']:
            send(build_vwap_exit_alert('LONG', entry, price))
            log('VWAP break - early exit alert sent')
            reset_state()
            return
        log('In LONG - entry $' + '{:,.2f}'.format(entry)
            + '  TP $' + '{:,.2f}'.format(tp_lvl)
            + '  SL $' + '{:,.2f}'.format(sl_lvl))
        return

    # IN A SHORT TRADE - monitor for exit
    if prev == 'SHORT' and entry and tp_lvl and sl_lvl:
        if price <= tp_lvl:
            send(build_tp_alert('SHORT', entry, price))
            log('TP hit - SHORT closed')
            reset_state()
            return
        if price >= sl_lvl:
            send(build_sl_alert('SHORT', entry, price))
            log('SL hit - SHORT closed')
            reset_state()
            return
        if sig['vwap_break_s']:
            send(build_vwap_exit_alert('SHORT', entry, price))
            log('VWAP break - early exit alert sent')
            reset_state()
            return
        log('In SHORT - entry $' + '{:,.2f}'.format(entry)
            + '  TP $' + '{:,.2f}'.format(tp_lvl)
            + '  SL $' + '{:,.2f}'.format(sl_lvl))
        return

    # NOT IN A TRADE - look for entries
    if sig['long_valid']:
        consec_l += 1
        consec_s  = 0
        log('LONG building: ' + str(consec_l) + '/3')
        if consec_l >= 3:
            send(build_long_alert(sig))
            log('LONG alert sent')
            save_state({'signal': 'LONG', 'entry': price,
                        'tp': sig['tp_l'], 'sl': sig['sl_l'],
                        'time': utc_now(), 'consec_l': consec_l, 'consec_s': 0})
        else:
            save_state({'signal': 'BUILDING', 'entry': None,
                        'tp': None, 'sl': None,
                        'time': utc_now(), 'consec_l': consec_l, 'consec_s': 0})

    elif sig['short_valid']:
        consec_s += 1
        consec_l  = 0
        log('SHORT building: ' + str(consec_s) + '/3')
        if consec_s >= 3:
            send(build_short_alert(sig))
            log('SHORT alert sent')
            save_state({'signal': 'SHORT', 'entry': price,
                        'tp': sig['tp_s'], 'sl': sig['sl_s'],
                        'time': utc_now(), 'consec_l': 0, 'consec_s': consec_s})
        else:
            save_state({'signal': 'BUILDING', 'entry': None,
                        'tp': None, 'sl': None,
                        'time': utc_now(), 'consec_l': 0, 'consec_s': consec_s})

    else:
        if prev in ('LONG', 'SHORT'):
            send(build_cleared_alert(sig, prev))
            log('Signal cleared')
        consec_l = 0
        consec_s = 0
        save_state({'signal': 'WAIT', 'entry': None,
                    'tp': None, 'sl': None,
                    'time': utc_now(), 'consec_l': 0, 'consec_s': 0})
        log('WAIT')

    log('Scan complete')

main()
