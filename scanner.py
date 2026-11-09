import json, sys, datetime as dt
import numpy as np, pandas as pd, yfinance as yf

PIVOT, K, MINB = 10, 6, 210
TFS = {"1d": ("2y", "1d"), "1h": ("3mo", "60m"), "30m": ("2mo", "30m"),
       "15m": ("1mo", "15m"), "5m": ("10d", "5m")}

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def f(x, d=2): return round(float(x), d) if np.isfinite(x) else 0

def rsi(c, n=14):
    d = c.diff(); up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High-df.Low, (df.High-pc).abs(), (df.Low-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def pivots(df):
    lo, hi = df.Low.values, df.High.values
    n = len(df); LL = np.full(n, np.nan); LH = np.full(n, np.nan); ll = lh = np.nan
    for i in range(n):
        j = i - PIVOT
        if j >= PIVOT:
            if lo[j] <= lo[j-PIVOT:j+PIVOT+1].min(): ll = lo[j]
            if hi[j] >= hi[j-PIVOT:j+PIVOT+1].max(): lh = hi[j]
        LL[i], LH[i] = ll, lh
    return LL, LH

def bars(df):
    if len(df) < MINB: return None
    c = df.Close; e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9)
    r, a = rsi(c), atr(df)
    vr = df.Volume / df.Volume.rolling(20).mean()
    LL, LH = pivots(df); n = len(df); out = []
    for i in range(n-K, n):
        ci, ai = float(c.iloc[i]), float(a.iloc[i])
        stop = max(LL[i], ci-2.5*ai) if (not np.isnan(LL[i]) and LL[i] < ci) else ci-2*ai
        risk = ci - stop
        if not np.isnan(LH[i]) and LH[i] > ci:
            t2 = LH[i] + 0.618*(LH[i] - (stop if np.isnan(LL[i]) else LL[i]))
        else: t2 = ci + 3*risk
        out.append(dict(t=str(df.index[i])[:16], c=f(ci), rsi=f(r.iloc[i], 1), vr=f(vr.iloc[i]),
            mx=int(macd.iloc[i] > sig.iloc[i] and macd.iloc[i-1] <= sig.iloc[i-1]),
            mp=int(macd.iloc[i] > sig.iloc[i]),
            tr=int(ci > e50.iloc[i] > e200.iloc[i]), e20=int(ci > e20.iloc[i]),
            brk=int(not np.isnan(LH[i]) and ci > LH[i] and c.iloc[i-1] <= LH[i]),
            sup=int(not np.isnan(LL[i]) and LL[i] <= ci <= LL[i] + 1.5*ai),
            s=f(stop), t1=f(ci + 2*risk), t2=f(t2)))
    return out

def fetch(tick, period, itv, size):
    fr = {}
    for i in range(0, len(tick), size):
        ch = tick[i:i+size]
        try:
            d = yf.download(ch, period=period, interval=itv, group_by="ticker",
                            auto_adjust=False, progress=False, threads=size > 10)
        except Exception:
            continue
        for t in ch:
            try:
                try: df = d[t]
                except Exception: df = d
                df = df.dropna(subset=["Close"])
                if len(df): fr[t] = df
            except Exception: pass
    return fr

def run(tf, items):
    period, itv = TFS[tf]; tick = [f"{c}.SR" for c, _ in items]
    fr = fetch(tick, period, itv, 40)
    miss = [t for t in tick if t not in fr]
    if miss: fr.update(fetch(miss, period, itv, 10))
    rows, failed, last = [], [], []
    for (code, name), t in zip(items, tick):
        try:
            df = fr.get(t); b = bars(df) if df is not None else None
        except Exception: b = None
        if b is None: failed.append(code); continue
        last.append(b[-1]["t"]); rows.append(dict(code=code, name=name, bars=b))
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=3))).strftime("%Y-%m-%d %H:%M")
    out = dict(updated=now, total=len(items), ok=len(rows), failed=failed,
               last=max(last) if last else "", rows=rows)
    with open(f"docs/data_{tf}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"{tf}: total={len(items)} ok={len(rows)} failed={failed}")

def main():
    items = []
    for line in open("tickers.txt", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            code, _, name = line.partition(",")
            code = code.strip().upper().replace("TADAWUL:", "").replace(".SR", "")
            items.append((code, name.strip() or code))
    for tf in (sys.argv[1:] or list(TFS)):
        try: run(tf, items)
        except Exception as e: print(tf, "failed:", e)

if __name__ == "__main__":
    main()
