"""Localización clásica RENALVISION, extraída del notebook IM_grup_v2_0.ipynb.

Conserva los parámetros y funciones del pipeline original; no incorpora la CNN.
"""
import base64
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import morphological_chan_vese
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu


# === CELDA 2: Parámetros ===
P = dict(
    # --- opcionales de contexto ---
    SEMILLA_RINON=None,   # (cx, cy, rx, ry) si la segmentación automática del riñón falla
    APICE=None,           # (x, y) del vértice del abanico (transductor convexo). None = haz vertical
    MM_POR_PIXEL=None,    # calibración si la conoces
    # --- riñón ---
    K_CLUSTERS=4, CV_ITER=60,
    # --- detección LoG (px a escala de 600 px de lado mayor) ---
    SIGMA_MIN=1.5, SIGMA_MAX=7.0, N_SIGMAS=6,
    K_LOG=3.0,            # umbral = media + K*desvío de la respuesta LoG dentro del riñón (más alto = más estricto)
    MAX_PICOS=30,
    # --- forma del cálculo ---
    AREA_MIN=10, AREA_MAX=800, SOLIDEZ_MIN=0.6, ALARGAMIENTO_MAX=3.5,
    # --- sombra acústica ---
    LARGO_SOMBRA=45, SOMBRA_IAS_MAX=0.70, CONTRASTE_MIN=1.5,
    # --- salida ---
    INCLUIR_POSIBLES=True, MAX_POSIBLES=5,
)

# === CELDA 3: Utilidades, sector del ultrasonido y SEGMENTACIÓN DEL RIÑÓN ===
def k_impar(v):
    v = int(round(v)); v = max(v, 3)
    return v if v % 2 == 1 else v + 1

def elipse_k(d):
    d = k_impar(d)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))

def casco_convexo(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    out = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(cnt)[:, 0, :], 1)
    return out.astype(bool)

def sector_us(gray, ESC):
    m = (cv2.GaussianBlur(gray, (0, 0), 2) > 12).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, elipse_k(15 * ESC))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    s = ndi.binary_fill_holes(m).astype(np.uint8)
    return cv2.erode(s, elipse_k(7 * ESC)).astype(bool)

def segmentar_rinon(gray, sector, ESC, P, lado=320):
    """Segmenta el riñón a menor resolución (más rápido) y devuelve la máscara a tamaño completo."""
    H, W = gray.shape
    f = lado / max(H, W)
    if f >= 1:
        return _segmentar_rinon(gray, sector, ESC, P)
    g = cv2.resize(gray, (int(round(W * f)), int(round(H * f))), interpolation=cv2.INTER_AREA)
    s = cv2.resize(sector.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    Pf = dict(P)
    if P["SEMILLA_RINON"] is not None:
        Pf["SEMILLA_RINON"] = tuple(v * f for v in P["SEMILLA_RINON"])
    m, origen, img_seg = _segmentar_rinon(g, s, ESC * f, Pf)
    if m is None:
        return None, origen, img_seg
    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) & sector
    return m, origen, cv2.resize(img_seg, (W, H))

def _segmentar_rinon(gray, sector, ESC, P):
    """K-means de intensidad -> candidato elíptico/central -> Chan-Vese morfológico -> casco convexo."""
    med = cv2.medianBlur(gray, 3)
    img_seg = cv2.fastNlMeansDenoising(med, None, h=12, templateWindowSize=7, searchWindowSize=21)
    img_seg = cv2.GaussianBlur(img_seg, (0, 0), 3 * ESC)
    area_sector = int(sector.sum())
    if area_sector < 2000:
        return None, "sector demasiado pequeño", img_seg
    ys_s, xs_s = np.nonzero(sector)
    cx_s, cy_s = xs_s.mean(), ys_s.mean()
    diag_s = np.hypot(np.ptp(xs_s), np.ptp(ys_s))

    if P["SEMILLA_RINON"] is not None:
        cx, cy, rx, ry = P["SEMILLA_RINON"]
        tmp = np.zeros(gray.shape, np.uint8)
        cv2.ellipse(tmp, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1, -1)
        init, origen = tmp.astype(bool), "semilla manual"
    else:
        pix = img_seg[sector].reshape(-1, 1).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
        _, etiq, _ = cv2.kmeans(pix, P["K_CLUSTERS"], None, crit, 3, cv2.KMEANS_PP_CENTERS)
        mapa = np.full(gray.shape, -1, np.int32)
        mapa[sector] = etiq.ravel()
        cands = []
        for c in range(P["K_CLUSTERS"]):
            mk = (mapa == c).astype(np.uint8)
            mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, elipse_k(7 * ESC))
            mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, elipse_k(11 * ESC))
            n, lab, _, _ = cv2.connectedComponentsWithStats(mk)
            for i in range(1, n):
                comp = ndi.binary_fill_holes(lab == i).astype(np.uint8)
                area = int(comp.sum())
                if not (0.02 * area_sector <= area <= 0.6 * area_sector):
                    continue
                cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = max(cnts, key=cv2.contourArea)
                if len(cnt) < 5:
                    continue
                solidez = area / max(cv2.contourArea(cv2.convexHull(cnt)), 1.0)
                (ex, ey), (d1, d2), _ = cv2.fitEllipse(cnt)
                if min(d1, d2) < 1:
                    continue
                el = area / (np.pi * d1 * d2 / 4)
                el = min(el, 1 / el) if el > 0 else 0
                alarg = max(d1, d2) / min(d1, d2)
                central = max(0.0, 1 - np.hypot(ex - cx_s, ey - cy_s) / (0.5 * diag_s))
                score = solidez * el * central * (1.0 if 1.2 <= alarg <= 3.5 else 0.5)
                cands.append((score, casco_convexo(comp)))
        if not cands:
            return None, "sin candidato a riñón", img_seg
        cands.sort(key=lambda t: -t[0])
        init, origen = cands[0][1], f"automática (score={cands[0][0]:.2f})"

    zona = cv2.dilate(init.astype(np.uint8), elipse_k(30 * ESC)).astype(bool) & sector
    ls = morphological_chan_vese(img_seg.astype(np.float32) / 255.0, num_iter=P["CV_ITER"],
                                 init_level_set=init.astype(np.uint8), smoothing=3).astype(bool)
    if (ls & init).sum() < 0.5 * init.sum():
        ls = ~ls
    ref = ls & zona
    n, lab = cv2.connectedComponents(ref.astype(np.uint8))
    if n > 1:
        solapes = [((lab == i) & init).sum() for i in range(1, n)]
        ref = lab == (1 + int(np.argmax(solapes)))
    else:
        ref = init
    ref = ndi.binary_fill_holes(ref)
    ref = cv2.morphologyEx(ref.astype(np.uint8), cv2.MORPH_CLOSE, elipse_k(15 * ESC)).astype(bool)
    if ref.sum() < 50:
        ref = init
    ref = casco_convexo(ref)
    ratio = ref.sum() / max(init.sum(), 1)
    if not (0.5 <= ratio <= 2.0):
        ref = init
    return (ref & sector), origen, img_seg

# === CELDA 5: MÉTODO MEJORADO — detección LoG, segmentación local, sombra, tamaño ===
def respuesta_log(th, ESC, P):
    sig = np.linspace(P["SIGMA_MIN"], P["SIGMA_MAX"], P["N_SIGMAS"]) * ESC
    pila = np.stack([-(s ** 2) * ndi.gaussian_laplace(th, s) for s in sig])   # LoG normalizado en escala
    return pila.max(0), sig[pila.argmax(0)]

def segmentar_local(img_det, rinon, y, x, s, ESC, P):
    H, W = img_det.shape
    r = int(max(10 * ESC, 4 * s))
    y0, y1, x0, x1 = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
    patch, mk = img_det[y0:y1, x0:x1], rinon[y0:y1, x0:x1]
    if mk.sum() < 30:
        return None
    v = patch[mk]
    if v.max() - v.min() < 10:
        return None
    t = max(threshold_otsu(v), np.median(v) + 25)
    b = ((patch >= t) & mk).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, elipse_k(3))
    _, lab = cv2.connectedComponents(b)
    py, px = y - y0, x - x0
    l = lab[py, px]
    if l == 0:
        sub = lab[max(0, py - 3):py + 4, max(0, px - 3):px + 4]
        ids = sub[sub > 0]
        if ids.size == 0:
            return None
        l = np.bincount(ids).argmax()
    comp = ndi.binary_fill_holes(lab == l)
    if comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any():
        return None                                  # toca el borde de la ventana: se fusionó con algo grande
    full = np.zeros((H, W), bool)
    full[y0:y1, x0:x1] = comp
    return full

def forma_ok(comp, ESC, P):
    area = int(comp.sum())
    if not (P["AREA_MIN"] * ESC ** 2 <= area <= P["AREA_MAX"] * ESC ** 2):
        return False
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(cnt))
    if hull <= 0 or cv2.contourArea(cnt) / hull < P["SOLIDEZ_MIN"]:
        return False
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    return min(rw, rh) >= 1 and max(rw, rh) / min(rw, rh) <= P["ALARGAMIENTO_MAX"]

def evaluar_sombra(img_det, sector, comp, ESC, P):
    """Perfil lateral de la franja posterior (a lo largo del haz). Devuelve IAS y ancho de la sombra."""
    H, W = img_det.shape
    ys, xs = np.nonzero(comp)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    d = np.array([0.0, 1.0]) if P["APICE"] is None else c - np.array(P["APICE"], float)
    nd = np.linalg.norm(d)
    d = d / nd if nd > 0 else np.array([0.0, 1.0])
    n = np.array([-d[1], d[0]])
    Pc = pts - c
    t, s = Pc @ d, Pc @ n
    w = max(s.max() - s.min() + 1, 6 * ESC)
    c0 = c + n * (s.min() + s.max()) / 2
    t0 = t.max() + 2
    t1 = t0 + P["LARGO_SOMBRA"] * ESC
    ts = np.arange(t0, t1, 1.0)
    ss = np.linspace(-3 * w, 3 * w, 61)
    TT, SS = np.meshgrid(ts, ss, indexing="ij")
    X, Y = c0[0] + d[0] * TT + n[0] * SS, c0[1] + d[1] * TT + n[1] * SS
    dentro = (X >= 0) & (X <= W - 1) & (Y >= 0) & (Y <= H - 1)
    Xc, Yc = np.clip(X, 0, W - 1), np.clip(Y, 0, H - 1)
    vals = ndi.map_coordinates(img_det, [Yc, Xc], order=1)
    sec = ndi.map_coordinates(sector.astype(np.float32), [Yc, Xc], order=0) > 0.5
    ok = dentro & sec
    cnt = ok.sum(0)
    prof = np.where(cnt >= 0.5 * len(ts), (vals * ok).sum(0) / np.maximum(cnt, 1), np.nan)
    central = np.abs(ss) <= 0.45 * w
    lateral = (np.abs(ss) >= 1.0 * w) & (np.abs(ss) <= 2.0 * w)
    if np.isnan(prof[central]).mean() > 0.5 or np.isfinite(prof[lateral]).sum() < 10:
        return None                                   # sombra fuera del sector: no evaluable
    br_som, br_ref = float(np.nanmean(prof[central])), float(np.nanmedian(prof[lateral]))
    if br_ref <= 1e-6:
        return None
    dip = np.nan_to_num(prof, nan=1e9) < 0.75 * br_ref
    ic = int(np.argmin(np.abs(ss)))
    ancho = 0.0
    if dip[ic]:
        l = r = ic
        while l > 0 and dip[l - 1]: l -= 1
        while r < len(ss) - 1 and dip[r + 1]: r += 1
        ancho = (r - l + 1) * (ss[1] - ss[0])
    poly = np.round(np.array([c0 + d * t0 + n * (-0.45 * w), c0 + d * t1 + n * (-0.45 * w),
                              c0 + d * t1 + n * (0.45 * w), c0 + d * t0 + n * (0.45 * w)])).astype(np.int32)
    return dict(br_ref=br_ref, br_som=br_som, ias=br_som / br_ref, ancho_sombra=float(ancho),
                poly=poly, ss=ss, prof=prof, w=w)

def analizar(gray, P):
    H, W = gray.shape
    ESC = max(H, W) / 600.0
    sector = sector_us(gray, ESC)
    rinon, origen, img_seg = segmentar_rinon(gray, sector, ESC, P)
    out = dict(gray=gray, ESC=ESC, sector=sector, rinon=rinon, origen=origen, img_seg=img_seg,
               resultados=[], R=None)
    if rinon is None:
        return out
    rinon_int = cv2.erode(rinon.astype(np.uint8), elipse_k(6 * ESC)).astype(bool)
    if rinon_int.sum() < 100:
        rinon_int = rinon
    img_det = cv2.GaussianBlur(cv2.medianBlur(gray, 3), (0, 0), 1.0).astype(np.float32)
    th = cv2.morphologyEx(img_det, cv2.MORPH_TOPHAT, elipse_k(25 * ESC))      # quita fondo suave (seno renal)
    R, S = respuesta_log(th, ESC, P)
    vals = R[rinon_int]
    umbral = vals.mean() + P["K_LOG"] * vals.std()
    picos = peak_local_max(np.where(rinon_int, R, 0), min_distance=max(2, int(4 * ESC)),
                           threshold_abs=umbral, exclude_border=False)
    picos = sorted(picos.tolist(), key=lambda p: -R[p[0], p[1]])[:P["MAX_PICOS"]]
    out.update(R=np.where(rinon_int, R, 0), umbral=umbral, n_picos=len(picos))

    aceptados = []
    for (y, x) in picos:
        comp = segmentar_local(img_det, rinon_int, y, x, S[y, x], ESC, P)
        if comp is None or not forma_ok(comp, ESC, P):
            continue
        if any((comp & a).sum() > 0.5 * comp.sum() for a in aceptados):
            continue
        aceptados.append(comp)
        sh = evaluar_sombra(img_det, sector, comp, ESC, P)
        if sh is None:
            continue
        contraste = float(img_det[comp].mean()) / sh["br_ref"]
        if contraste >= P["CONTRASTE_MIN"] and sh["ias"] <= P["SOMBRA_IAS_MAX"]:
            clase = "CONFIRMADO"
        elif P["INCLUIR_POSIBLES"] and contraste >= P["CONTRASTE_MIN"] * 1.3:
            clase = "POSIBLE"
        else:
            continue
        ys, xs = np.nonzero(comp)
        out["resultados"].append(dict(
            clase=clase, comp=comp, centro=(float(xs.mean()), float(ys.mean())), area=int(comp.sum()),
            diam_px=float(2 * np.sqrt(comp.sum() / np.pi)), contraste=contraste, R=float(R[y, x]), **sh))

    conf = [r for r in out["resultados"] if r["clase"] == "CONFIRMADO"]
    posi = sorted([r for r in out["resultados"] if r["clase"] != "CONFIRMADO"], key=lambda r: -r["R"])[:P["MAX_POSIBLES"]]
    res = sorted(conf, key=lambda r: r["ias"]) + posi
    for k, r in enumerate(res, 1):
        r["id"] = k
    out["resultados"] = res
    return out

def hay_calculo(out, modo="estricto"):
    if modo == "estricto":
        return any(r["clase"] == "CONFIRMADO" for r in out["resultados"])
    return len(out["resultados"]) > 0

# === CELDA 6: Visualización y reporte ===
def renderizar_imagen(out):
    gray = out["gray"]
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ov = base.copy()
    if out["rinon"] is not None:
        c, _ = cv2.findContours(out["rinon"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, c, -1, (0, 255, 255), 2)
    for r in out["resultados"]:
        col = (0, 255, 0) if r["clase"] == "CONFIRMADO" else (0, 165, 255)
        cc, _ = cv2.findContours(r["comp"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cc, -1, col, 2)
        cv2.polylines(ov, [r["poly"]], True, (0, 0, 255), 1)
        cv2.putText(ov, f"#{r['id']}", (int(r["centro"][0]) + 6, int(r["centro"][1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * max(out["ESC"], 0.8), col, 1)
    # El mismo overlay que mostrar(out, P) guarda con cv2.imwrite().
    valido, png = cv2.imencode(".png", ov)
    if not valido:
        raise RuntimeError("No se pudo codificar la imagen procesada")
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")



def procesar_localizacion(gray):
    """Procesa una ecografía en gris; devuelve regiones y overlay PNG como data URL.

    `localization_positive` conserva el modo estricto de `hay_calculo`: exige CONFIRMADO.
    `detections` cuenta tanto CONFIRMADO como POSIBLE del pipeline original.
    """
    out = analizar(gray, P)
    return {
        "detections": len(out["resultados"]),
        "localization_positive": bool(hay_calculo(out)),
        "processed_image": renderizar_imagen(out),
    }
