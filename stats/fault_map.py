"""Real map of two neighbouring faults (GEM Global Active Faults Database) with cartopy.
Run in Colab (needs internet + cartopy). Writes figs/fault_map.pdf/.png and out_fault_pair.json.
Change REGION / NAME1 / NAME2 to choose another pair of faults (the printed table lists candidates)."""
import json, os, sys, urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)
URLS = ["https://raw.githubusercontent.com/cossatot/gem-global-active-faults/master/geojson/gem_active_faults.geojson",
        "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults.geojson"]
# ---- user choices ------------------------------------------------------------------------------------
REGION = (43.0, 51.0, 35.5, 40.5)      # lon_min, lon_max, lat_min, lat_max  (north-west Iran, around Tabriz)
NAME1 = os.environ.get("FAULT1", "Tabriz")  # substring of the name of fault 1 (None = longest fault in the box)
NAME2 = os.environ.get("FAULT2", "")        # substring of fault 2 ("" = nearest other fault longer than MINLEN)
MINLEN = 40.0                          # km
# ------------------------------------------------------------------------------------------------------


def haversine(lon1, lat1, lon2, lat2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def load():
    fn = os.path.join(DATA, "gem_active_faults.geojson")
    if not os.path.exists(fn):
        for u in URLS:
            try:
                print("downloading", u)
                urllib.request.urlretrieve(u, fn)
                break
            except Exception as e:
                print("failed:", e)
    return json.load(open(fn))


def lines_of(geom):
    if geom["type"] == "LineString":
        return [np.array(geom["coordinates"])[:, :2]]
    if geom["type"] == "MultiLineString":
        return [np.array(c)[:, :2] for c in geom["coordinates"]]
    return []


gj = load()
lo0, lo1, la0, la1 = REGION
F = []
for ft in gj["features"]:
    ls = lines_of(ft["geometry"])
    ls = [l for l in ls if len(l) > 1 and (l[:, 0] > lo0).any() and (l[:, 0] < lo1).any() and (l[:, 1] > la0).any() and (l[:, 1] < la1).any()]
    if not ls:
        continue
    pr = ft.get("properties", {})
    name = str(pr.get("name") or pr.get("catalog_name") or pr.get("fault_name") or "unnamed")
    length = sum(haversine(l[:-1, 0], l[:-1, 1], l[1:, 0], l[1:, 1]).sum() for l in ls)
    F.append(dict(name=name, lines=ls, length=length, slip=pr.get("slip_type", "")))
print(f"{len(F)} fault traces in the box; longest ones:")
for f in sorted(F, key=lambda f: -f["length"])[:25]:
    print(f"  {f['name'][:50]:50s} {f['length']:7.1f} km  {f['slip']}")


def pts(f):
    p = np.vstack(f["lines"])
    return p[:: max(1, len(p) // 1500)]


def mind(f, g):
    a, b = pts(f), pts(g)
    D = haversine(a[:, None, 0], a[:, None, 1], b[None, :, 0], b[None, :, 1])
    i, j = np.unravel_index(D.argmin(), D.shape)
    return D[i, j], a[i], b[j]


c1 = [f for f in F if NAME1 and NAME1.lower() in f["name"].lower()] or sorted(F, key=lambda f: -f["length"])[:1]
f1 = max(c1, key=lambda f: f["length"])
if NAME2:
    f2 = max([f for f in F if NAME2.lower() in f["name"].lower()], key=lambda f: f["length"])
else:
    cand = [f for f in F if f is not f1 and f["name"] != f1["name"] and f["length"] >= MINLEN]
    f2 = min(cand, key=lambda f: mind(f1, f)[0])
d, p1, p2 = mind(f1, f2)
print(f"\nFault 1: {f1['name']} ({f1['length']:.0f} km)\nFault 2: {f2['name']} ({f2['length']:.0f} km)\nclosest approach d = {d:.1f} km")
json.dump(dict(fault1=f1["name"], length1_km=f1["length"], fault2=f2["name"], length2_km=f2["length"], d_km=float(d),
               source="GEM Global Active Faults Database (Styron & Pagani, 2020)"),
          open(os.path.join(HERE, "out_fault_pair.json"), "w"), indent=1)

fig = plt.figure(figsize=(7.4, 5.6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lo0, lo1, la0, la1], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="#f2efe6")
ax.add_feature(cfeature.OCEAN, facecolor="#dbe9f4")
ax.add_feature(cfeature.LAKES, facecolor="#dbe9f4", edgecolor="#7aa6c9", lw=0.4)
ax.add_feature(cfeature.BORDERS, lw=0.6, ls="--", edgecolor="#777777")
ax.add_feature(cfeature.COASTLINE, lw=0.6)
gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.6)
gl.top_labels = gl.right_labels = False
for f in F:
    for l in f["lines"]:
        ax.plot(l[:, 0], l[:, 1], color="#999999", lw=0.6, transform=ccrs.PlateCarree(), zorder=2)
for f, col, lab in ((f1, "#0072B2", "Fault 1"), (f2, "#D55E00", "Fault 2")):
    for k, l in enumerate(f["lines"]):
        ax.plot(l[:, 0], l[:, 1], color=col, lw=2.6, transform=ccrs.PlateCarree(), zorder=4,
                label=f"{lab}: {f['name']} ({f['length']:.0f} km)" if k == 0 else None)
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="k", ls="--", lw=1.2, transform=ccrs.PlateCarree(), zorder=5)
ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, f"  $d\\approx{d:.0f}$ km", transform=ccrs.PlateCarree(), zorder=6,
        fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
for nm, lon, lat in (("Tabriz", 46.29, 38.08), ("Tehran", 51.39, 35.69), ("Urmia", 45.07, 37.55)):
    if lo0 < lon < lo1 and la0 < lat < la1:
        ax.plot(lon, lat, "o", color="k", ms=3.5, transform=ccrs.PlateCarree(), zorder=6)
        ax.text(lon + 0.07, lat + 0.05, nm, fontsize=7.5, transform=ccrs.PlateCarree(), zorder=6)
ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
ax.set_title("Two neighbouring active faults (real traces)  --  GEM Global Active Faults Database", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fault_map.pdf"))
fig.savefig(os.path.join(FIG, "fault_map.png"), dpi=170)
print("saved figs/fault_map.pdf / .png")
