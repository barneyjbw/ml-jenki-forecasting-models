import pandas as pd, numpy as np, sys, os, warnings
warnings.filterwarnings('ignore')
import geopandas as gpd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
from pathlib import Path
from branca.colormap import LinearColormap
from scipy.stats import pearsonr
sys.stdout.reconfigure(encoding='utf-8')

DATA  = Path(r'C:\Users\ibarn\Documents\GitHub\ml-jenki-forecasting-models\data\source-data')
OUT   = Path(r'C:\Users\ibarn\Documents\GitHub\ml-jenki-forecasting-models\experiments')
PLOTS = Path(r'C:\Users\ibarn\Documents\GitHub\ml-jenki-forecasting-models\plots')
SHP   = Path(r'C:\Users\ibarn\AppData\Local\Temp\oa_shapes\OA_2021_EW_BFE_V9.shp')

# ── 1. Compute composite index for 5 training stores ─────────────────────────
persona = pd.read_csv(r'C:\Users\ibarn\Downloads\persona_oa_props_impactScore_comm.csv').set_index('code')

STORE_OAS     = {'borough':'E00019779','battersea':'E00183178','canary_wharf':'E00167122',
                 'covent_garden':'E00004529','spitalfields':'E00021686'}
STORE_REVENUE = {'borough':2329.35,'battersea':2410.99,'canary_wharf':1961.93,
                 'covent_garden':2254.21,'spitalfields':3751.40}
STORE_POP     = {'borough':331,'battersea':289,'canary_wharf':458,'covent_garden':278,'spitalfields':644}

rows = []
for store, oa in STORE_OAS.items():
    b6 = persona.loc[oa, 'Bombe 6']
    b7 = persona.loc[oa, 'Bombe 7']
    pop = STORE_POP[store]
    raw_score = (b6 + b7) * pop
    rows.append({'store': store, 'bombe_6': b6, 'bombe_7': b7,
                 'population': pop, 'raw_score': raw_score,
                 'mean_daily_revenue': STORE_REVENUE[store]})

train_df = pd.DataFrame(rows).set_index('store')

# ── 2. Anchor: linear scale raw_score → £ revenue ────────────────────────────
# Fit: revenue = a * raw_score + b  (simple OLS, 5 points)
from numpy.polynomial import polynomial as P
x = train_df['raw_score'].values
y = train_df['mean_daily_revenue'].values
coeffs = np.polyfit(x, y, 1)  # [slope, intercept]
a, b = coeffs
train_df['predicted_revenue'] = a * x + b

r, _ = pearsonr(x, y)
residuals = train_df['predicted_revenue'] - y
mae = np.mean(np.abs(residuals))

print('Training stores:')
print(train_df[['bombe_6','bombe_7','population','raw_score','mean_daily_revenue','predicted_revenue']].round(2).to_string())
print(f'\nAnchor: revenue = {a:.1f} * score + {b:.1f}')
print(f'Pearson r (raw_score vs revenue): {r:.3f}')
print(f'In-sample MAE: £{mae:.0f}')

# ── 3. Feature importance / validation plot ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: bar chart of raw score components per store
ax = axes[0]
stores = train_df.index.tolist()
x_pos = np.arange(len(stores))
b6_contrib = train_df['bombe_6'] * train_df['population']
b7_contrib = train_df['bombe_7'] * train_df['population']
bars1 = ax.bar(x_pos, b6_contrib, label='Bombe 6 × Population', color='#2171b5')
bars2 = ax.bar(x_pos, b7_contrib, bottom=b6_contrib, label='Bombe 7 × Population', color='#6baed6')
ax.set_xticks(x_pos)
ax.set_xticklabels([s.replace('_','\n').title() for s in stores], fontsize=8)
ax.set_ylabel('Composite Score')
ax.set_title('Score Components per Store\n(Bombe 6 + Bombe 7) × Population', fontweight='bold')
ax.legend(fontsize=9)

# Right: score vs actual revenue
ax2 = axes[1]
store_colors = {'borough':'#4e79a7','battersea':'#f28e2b','canary_wharf':'#e15759',
                'covent_garden':'#76b7b2','spitalfields':'#59a14f'}
for store in stores:
    ax2.scatter(train_df.loc[store,'raw_score'], train_df.loc[store,'mean_daily_revenue'],
                color=store_colors[store], s=120, zorder=3, label=store.replace('_',' ').title())
xs = np.linspace(x.min()*0.9, x.max()*1.1, 100)
ax2.plot(xs, a*xs + b, 'k--', linewidth=1.2, label='Anchored fit')
ax2.set_xlabel('Composite Score  (Bombe 6 + 7) × Population', fontsize=10)
ax2.set_ylabel('Actual Mean Daily Revenue (£)', fontsize=10)
ax2.set_title(f'Score vs Revenue (r={r:.2f})\nAnchored to store actuals', fontweight='bold')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS / 'feature_importance.png', dpi=150, bbox_inches='tight')
print('\nPlot saved.')

# ── 4. Score all London OAs ───────────────────────────────────────────────────
print('Scoring all London OAs...')
oa_rgn = pd.read_csv(DATA / 'OA21_RGN22_LU.csv')
london_oas = set(oa_rgn[oa_rgn['rgn22nm'] == 'London']['oa21cd'].unique())
london_persona = persona[persona.index.isin(london_oas)][['Bombe 6','Bombe 7']].copy()

sape = pd.read_excel(
    os.path.join(os.environ['TEMP'], 'sape.xlsx'),
    sheet_name='Mid-2022 OA 2021', header=3, usecols=['OA 2021 Code','Total']
)
sape.columns = ['oa21cd','population']
sape = sape.dropna(subset=['oa21cd'])
sape['population'] = pd.to_numeric(sape['population'], errors='coerce')
sape_idx = sape.set_index('oa21cd')

scores = london_persona.copy()
scores['population'] = sape_idx.reindex(scores.index)['population'].fillna(sape_idx['population'].median())
scores['raw_score'] = (scores['Bombe 6'] + scores['Bombe 7']) * scores['population']
scores['predicted_revenue'] = (a * scores['raw_score'] + b).clip(lower=300)
scores['predicted_revenue'] = scores['predicted_revenue'].clip(
    upper=scores['predicted_revenue'].quantile(0.995)
)

print(f'Scored {len(scores):,} OAs')
print(scores['predicted_revenue'].describe().round(0))

scores.reset_index().rename(columns={'code':'oa21cd'})[['oa21cd','predicted_revenue']].to_parquet(
    OUT / 'london_oa_scores_v2.parquet', index=False
)

# ── 5. London-only shapefile + map ────────────────────────────────────────────
print('Loading shapefile...')
gdf = gpd.read_file(SHP)
gdf_london = gdf[gdf['OA21CD'].isin(london_oas)].copy()
gdf_london = gdf_london.to_crs('EPSG:4326')
gdf_london['geometry'] = gdf_london['geometry'].simplify(tolerance=0.0001, preserve_topology=True)
gdf_london = gdf_london.merge(
    scores[['predicted_revenue']].reset_index().rename(columns={'code':'OA21CD','index':'OA21CD'}),
    on='OA21CD', how='left'
)
# fix merge key
if 'predicted_revenue' not in gdf_london.columns:
    score_map = scores['predicted_revenue'].to_dict()
    gdf_london['predicted_revenue'] = gdf_london['OA21CD'].map(score_map)

med = scores['predicted_revenue'].median()
gdf_london['predicted_revenue'] = gdf_london['predicted_revenue'].fillna(med)
gdf_london['predicted_revenue_fmt'] = gdf_london['predicted_revenue'].round(0).astype(int)
print(f'OAs in map: {len(gdf_london):,}')

print('Rendering map...')
m = folium.Map(location=[51.505, -0.09], zoom_start=10,
               tiles='CartoDB positron', prefer_canvas=True)

vmin = scores['predicted_revenue'].quantile(0.05)
vmax = scores['predicted_revenue'].quantile(0.95)
colormap = LinearColormap(
    colors=['#deebf7','#9ecae1','#4292c6','#2171b5','#084594'],
    vmin=vmin, vmax=vmax,
    caption='Revenue Potential Index (£ anchored)'
)

folium.GeoJson(
    gdf_london[['geometry','OA21CD','predicted_revenue_fmt']].to_json(),
    style_function=lambda feat: {
        'fillColor':   colormap(float(feat['properties']['predicted_revenue_fmt'] or vmin)),
        'color':       'none',
        'fillOpacity': 0.75,
        'weight':      0,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['OA21CD','predicted_revenue_fmt'],
        aliases=['OA Code','Revenue Potential (£)'],
        localize=True,
    ),
).add_to(m)
colormap.add_to(m)

STORE_LATLNG = {
    'Borough':(51.5055,-0.0908),'Battersea':(51.4839,-0.1445),
    'Canary Wharf':(51.5054,-0.0235),'Covent Garden':(51.5129,-0.1243),
    'Spitalfields':(51.5194,-0.0749),
}
STORE_REV = {'Borough':2329,'Battersea':2411,'Canary Wharf':1962,'Covent Garden':2254,'Spitalfields':3751}
for name, (lat, lon) in STORE_LATLNG.items():
    folium.CircleMarker(
        location=[lat, lon], radius=8,
        color='#ff4444', fill=True, fill_color='#ff4444', fill_opacity=0.9,
        tooltip=f'<b>{name}</b><br>Actual avg: £{STORE_REV[name]:,}/day',
    ).add_to(m)
    folium.Marker(
        location=[lat+0.003, lon],
        icon=folium.DivIcon(
            html=f'<div style="font-size:10px;font-weight:bold;color:#cc0000;white-space:nowrap">{name}</div>'
        ),
    ).add_to(m)

out_path = PLOTS / 'london_oa_revenue_map.html'
m.save(str(out_path))
print(f'Map saved: {out_path}')
