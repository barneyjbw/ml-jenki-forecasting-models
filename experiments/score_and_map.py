"""
Score all London OAs with predicted mean daily revenue and render Folium choropleth map.
Run after london_oa_visitor_persona.parquet is ready.
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, pickle, sys, os
import geopandas as gpd
import folium
from pathlib import Path
from branca.colormap import LinearColormap
sys.stdout.reconfigure(encoding='utf-8')

DATA  = Path(r'C:\Users\ibarn\Documents\GitHub\ml-jenki-forecasting-models\data\source-data')
OUT   = Path(r'C:\Users\ibarn\Documents\GitHub\ml-jenki-forecasting-models\experiments')
PLOTS = Path(r'C:\Users\ibarn\Documents\GitHub\ml-jenki-forecasting-models\plots')
SHP   = Path(r'C:\Users\ibarn\AppData\Local\Temp\oa_shapes\OA_2021_EW_BFE_V9.shp')

# ── Load artefacts ────────────────────────────────────────────────────────────
ridge     = pickle.load(open(OUT / 'ridge_model.pkl','rb'))
scaler    = pickle.load(open(OUT / 'ridge_scaler.pkl','rb'))
feat_cols = pickle.load(open(OUT / 'ridge_feature_cols.pkl','rb'))
visitor_df = pd.read_parquet(OUT / 'london_oa_visitor_persona.parquet')
sape = pd.read_excel(
    os.path.join(os.environ['TEMP'], 'sape.xlsx'),
    sheet_name='Mid-2022 OA 2021', header=3, usecols=['OA 2021 Code','Total']
)
sape.columns = ['oa21cd','population']
sape = sape.dropna(subset=['oa21cd'])
sape['population'] = pd.to_numeric(sape['population'], errors='coerce')

print(f'Visitor persona OAs: {len(visitor_df):,}')
print(f'OAs with full data: {visitor_df.dropna().shape[0]:,}')

# ── Assemble scoring dataframe ────────────────────────────────────────────────
visitor_cols = [c for c in feat_cols if c.startswith('visitor_')]
month_dummy_cols = [c for c in feat_cols if c.startswith('m_')]

scores = visitor_df.dropna().copy()
scores = scores.merge(sape[['oa21cd','population']], on='oa21cd', how='left')
scores['population'] = scores['population'].fillna(scores['population'].median())

# mean_dow: use overall mean from training panel
panel = pd.read_csv(OUT / 'panel.csv')
mean_dow_val = panel['mean_dow'].mean()
scores['mean_dow'] = mean_dow_val

# Month dummies: set to 0 (predicting average, not a specific month)
for col in month_dummy_cols:
    scores[col] = 0.0

# Build feature matrix in correct order
X = scores[feat_cols].values.astype(float)
X_sc = scaler.transform(X)
scores['predicted_revenue'] = ridge.predict(X_sc)

# Clip to sensible range (floor at £500, cap at 99th pct)
floor = 500.0
cap   = scores['predicted_revenue'].quantile(0.99)
scores['predicted_revenue'] = scores['predicted_revenue'].clip(lower=floor, upper=cap)

print(f'\nPredicted revenue stats:')
print(scores['predicted_revenue'].describe().round(0))

# ── Load shapefile and join ───────────────────────────────────────────────────
print('\nLoading shapefile...')
gdf = gpd.read_file(SHP)
oa_rgn = pd.read_csv(DATA / 'OA21_RGN22_LU.csv')
london_oas = set(oa_rgn[oa_rgn['rgn22nm'] == 'London']['oa21cd'].unique())
gdf_london = gdf[gdf['OA21CD'].isin(london_oas)].copy()
gdf_london = gdf_london.to_crs('EPSG:4326')
gdf_london = gdf_london.merge(
    scores[['oa21cd','predicted_revenue','population']],
    left_on='OA21CD', right_on='oa21cd', how='left'
)
coverage = gdf_london['predicted_revenue'].notna().sum()
print(f'OAs with predictions: {coverage:,}/{len(gdf_london):,} ({coverage/len(gdf_london)*100:.1f}%)')

# Fill missing with median
med = gdf_london['predicted_revenue'].median()
gdf_london['predicted_revenue'] = gdf_london['predicted_revenue'].fillna(med)

# ── Build Folium map ──────────────────────────────────────────────────────────
print('Rendering map...')
m = folium.Map(
    location=[51.505, -0.09],
    zoom_start=11,
    tiles='CartoDB positron',
    prefer_canvas=True,
)

# Blue colormap: light → dark
colormap = LinearColormap(
    colors=['#deebf7','#9ecae1','#4292c6','#2171b5','#08306b'],
    vmin=gdf_london['predicted_revenue'].quantile(0.05),
    vmax=gdf_london['predicted_revenue'].quantile(0.95),
    caption='Predicted Mean Daily Revenue (£)'
)

folium.GeoJson(
    gdf_london[['geometry','OA21CD','predicted_revenue','population']].to_json(),
    style_function=lambda feat: {
        'fillColor':   colormap(feat['properties']['predicted_revenue']
                                if feat['properties']['predicted_revenue'] else med),
        'color':       'none',
        'fillOpacity': 0.75,
        'weight':      0,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['OA21CD','predicted_revenue','population'],
        aliases=['OA Code','Est. Daily Revenue (£)','Population'],
        localize=True,
    ),
).add_to(m)

colormap.add_to(m)

# Store markers
STORE_LATLNG = {
    'Borough':       (51.5055, -0.0908),
    'Battersea':     (51.4839, -0.1445),
    'Canary Wharf':  (51.5054, -0.0235),
    'Covent Garden': (51.5129, -0.1243),
    'Spitalfields':  (51.5194, -0.0749),
}
STORE_REVENUE = {
    'Borough':2329,'Battersea':2411,'Canary Wharf':1962,'Covent Garden':2254,'Spitalfields':3751
}
for name, (lat, lon) in STORE_LATLNG.items():
    folium.CircleMarker(
        location=[lat, lon],
        radius=8,
        color='#ff4444',
        fill=True,
        fill_color='#ff4444',
        fill_opacity=0.9,
        tooltip=f'{name} — actual avg: £{STORE_REVENUE[name]:,}/day',
    ).add_to(m)
    folium.Marker(
        location=[lat + 0.003, lon],
        icon=folium.DivIcon(html=f'<div style="font-size:10px;font-weight:bold;color:#cc0000;white-space:nowrap">{name}</div>'),
    ).add_to(m)

out_path = PLOTS / 'london_oa_revenue_map.html'
m.save(str(out_path))
print(f'\nMap saved: {out_path}')
print('Done.')
