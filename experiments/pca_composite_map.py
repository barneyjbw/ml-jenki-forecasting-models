import pandas as pd, numpy as np, sys, os, warnings
warnings.filterwarnings('ignore')
import geopandas as gpd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
from pathlib import Path
from branca.colormap import LinearColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
sys.stdout.reconfigure(encoding='utf-8')

DATA  = Path(r'C:\Users\ibarn\Documents\GitHub\ml-jenki-forecasting-models\data\source-data')
OUT   = Path(r'C:\Users\ibarn\Documents\GitHub\ml-jenki-forecasting-models\experiments')
PLOTS = Path(r'C:\Users\ibarn\Documents\GitHub\ml-jenki-forecasting-models\plots')
SHP   = Path(r'C:\Users\ibarn\AppData\Local\Temp\oa_shapes\OA_2021_EW_BFE_V9.shp')

# ── 1. Load all London OA personas (resident + visitor) ───────────────────────
print('Loading data...')
oa_rgn = pd.read_csv(DATA / 'OA21_RGN22_LU.csv')
london_oas = set(oa_rgn[oa_rgn['rgn22nm'] == 'London']['oa21cd'].unique())

resident = pd.read_csv(
    r'C:\Users\ibarn\Downloads\persona_oa_props_impactScore_comm.csv'
).set_index('code')
resident = resident[resident.index.isin(london_oas)]
resident.columns = [f'res_{c}' for c in resident.columns]

visitor = pd.read_parquet(OUT / 'london_oa_visitor_persona.parquet').set_index('oa21cd')
visitor.columns = [f'vis_{c}' for c in visitor.columns]

# Combine — inner join so we only score OAs with both
combined = resident.join(visitor, how='inner')
print(f'OAs with both resident + visitor personas: {len(combined):,}')
print(f'Features: {combined.shape[1]}  ({list(combined.columns)})')

# ── 2. PCA across all London OAs ──────────────────────────────────────────────
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(combined.values)

pca = PCA(n_components=5)
pca.fit(X_scaled)

print(f'\nVariance explained by each PC:')
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f'  PC{i+1}: {var*100:.1f}%')

# PC1 loadings — show which personas drive it
loadings = pd.Series(pca.components_[0], index=combined.columns).sort_values(ascending=False)
print(f'\nPC1 loadings (top 5 positive):')
print(loadings.head(5).round(3).to_string())
print(f'PC1 loadings (top 5 negative):')
print(loadings.tail(5).round(3).to_string())

# Project all London OAs
pcs = pca.transform(X_scaled)
combined['PC1'] = pcs[:, 0]
combined['PC2'] = pcs[:, 1]

# ── 3. Load population ────────────────────────────────────────────────────────
sape = pd.read_excel(
    os.path.join(os.environ['TEMP'], 'sape.xlsx'),
    sheet_name='Mid-2022 OA 2021', header=3, usecols=['OA 2021 Code','Total']
)
sape.columns = ['oa21cd','population']
sape = sape.dropna(subset=['oa21cd'])
sape['population'] = pd.to_numeric(sape['population'], errors='coerce')
sape_idx = sape.set_index('oa21cd')
combined['population'] = sape_idx.reindex(combined.index)['population'].fillna(sape_idx['population'].median())

# PC1 runs suburban→urban; flip so high score = urban (inner London) persona
# Check: if Bombe 4 (suburban) loads positively, PC1 is suburban axis → negate
bombe4_loading = loadings.get('res_Bombe 4', 0)
if bombe4_loading > 0:
    combined['PC1'] = -combined['PC1']
    print('(PC1 sign flipped: now high PC1 = urban/Bombe 6+7 profile)')

combined['composite'] = combined['PC1'] * combined['population']

# ── 4. Anchor to 5 store revenues ────────────────────────────────────────────
STORE_OAS     = {'borough':'E00019779','battersea':'E00183178','canary_wharf':'E00167122',
                 'covent_garden':'E00004529','spitalfields':'E00021686'}
STORE_REVENUE = {'borough':2329.35,'battersea':2410.99,'canary_wharf':1961.93,
                 'covent_garden':2254.21,'spitalfields':3751.40}

store_rows = []
for store, oa in STORE_OAS.items():
    if oa in combined.index:
        store_rows.append({
            'store': store,
            'oa21cd': oa,
            'PC1': combined.loc[oa, 'PC1'],
            'population': combined.loc[oa, 'population'],
            'composite': combined.loc[oa, 'composite'],
            'mean_daily_revenue': STORE_REVENUE[store],
        })

train_df = pd.DataFrame(store_rows).set_index('store')
print('\nTraining stores:')
print(train_df[['PC1','population','composite','mean_daily_revenue']].round(2).to_string())

x = train_df['composite'].values
y = train_df['mean_daily_revenue'].values
r, _ = pearsonr(x, y)
print(f'\nPearson r (composite vs revenue): {r:.3f}')

# Fit anchor
coeffs = np.polyfit(x, y, 1)
a, b = coeffs
train_df['predicted_revenue'] = a * x + b
mae = np.mean(np.abs(train_df['predicted_revenue'] - y))
print(f'Anchor: revenue = {a:.2f} * composite + {b:.0f}')
print(f'In-sample MAE: £{mae:.0f}')
print('\nStore predictions:')
for store in train_df.index:
    act = train_df.loc[store,'mean_daily_revenue']
    pred = train_df.loc[store,'predicted_revenue']
    print(f'  {store:15s}  actual=£{act:.0f}  pred=£{pred:.0f}  err={act-pred:+.0f}')

# ── 5. Feature importance / PCA plot ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PC1 loadings
ax = axes[0]
top_loadings = pd.concat([loadings.head(7), loadings.tail(7)])
colors = ['#2171b5' if v > 0 else '#d6604d' for v in top_loadings]
ax.barh(top_loadings.index.str.replace('res_','R: ').str.replace('vis_','V: '),
        top_loadings.values, color=colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('PC1 Loadings\n(which personas define the axis)', fontweight='bold')
ax.set_xlabel('Loading')

# Composite vs revenue scatter
ax2 = axes[1]
store_colors = {'borough':'#4e79a7','battersea':'#f28e2b','canary_wharf':'#e15759',
                'covent_garden':'#76b7b2','spitalfields':'#59a14f'}
for store in train_df.index:
    ax2.scatter(train_df.loc[store,'composite'], train_df.loc[store,'mean_daily_revenue'],
                color=store_colors[store], s=120, zorder=3, label=store.replace('_',' ').title())
xs = np.linspace(x.min()*0.9, x.max()*1.1, 100)
ax2.plot(xs, a*xs + b, 'k--', linewidth=1.2)
ax2.set_xlabel('PC1 × Population (composite)', fontsize=10)
ax2.set_ylabel('Mean Daily Revenue (£)', fontsize=10)
ax2.set_title(f'Composite vs Revenue\nr={r:.2f}, MAE=£{mae:.0f}', fontweight='bold')
ax2.legend(fontsize=8)

# Variance explained scree
ax3 = axes[2]
ax3.bar(range(1, 6), pca.explained_variance_ratio_ * 100, color='#4292c6', edgecolor='white')
ax3.set_xlabel('Principal Component')
ax3.set_ylabel('Variance Explained (%)')
ax3.set_title('PCA Scree Plot\n(across all London OAs)', fontweight='bold')
for i, v in enumerate(pca.explained_variance_ratio_):
    ax3.text(i+1, v*100+0.3, f'{v*100:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS / 'feature_importance.png', dpi=150, bbox_inches='tight')
print('\nPlot saved.')

# ── 6. Score all London OAs → £ ──────────────────────────────────────────────
combined['predicted_revenue'] = (a * combined['composite'] + b).clip(lower=300)
combined['predicted_revenue'] = combined['predicted_revenue'].clip(
    upper=combined['predicted_revenue'].quantile(0.995)
)
print(f'\nAll London OA predicted revenue:')
print(combined['predicted_revenue'].describe().round(0))

# ── 7. London shapefile + Folium map ─────────────────────────────────────────
print('\nLoading shapefile...')
gdf = gpd.read_file(SHP)
gdf_london = gdf[gdf['OA21CD'].isin(london_oas)].copy()
gdf_london = gdf_london.to_crs('EPSG:4326')
gdf_london['geometry'] = gdf_london['geometry'].simplify(tolerance=0.0001, preserve_topology=True)

score_map = combined['predicted_revenue'].to_dict()
gdf_london['predicted_revenue'] = gdf_london['OA21CD'].map(score_map)
med = combined['predicted_revenue'].median()
gdf_london['predicted_revenue'] = gdf_london['predicted_revenue'].fillna(med)
gdf_london['rev_fmt'] = gdf_london['predicted_revenue'].round(0).astype(int)

coverage = gdf_london['OA21CD'].isin(combined.index).sum()
print(f'OAs with predictions: {coverage:,}/{len(gdf_london):,}')

print('Rendering map...')
m = folium.Map(location=[51.505, -0.09], zoom_start=10,
               tiles='CartoDB positron', prefer_canvas=True)

vmin = combined['predicted_revenue'].quantile(0.05)
vmax = combined['predicted_revenue'].quantile(0.95)
colormap = LinearColormap(
    colors=['#deebf7','#9ecae1','#4292c6','#2171b5','#084594'],
    vmin=vmin, vmax=vmax,
    caption='Predicted Revenue Potential (£/day)'
)

folium.GeoJson(
    gdf_london[['geometry','OA21CD','rev_fmt']].to_json(),
    style_function=lambda feat: {
        'fillColor':   colormap(float(feat['properties']['rev_fmt'] or vmin)),
        'color':       'none',
        'fillOpacity': 0.75,
        'weight':      0,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['OA21CD','rev_fmt'],
        aliases=['OA Code','Revenue Potential (£/day)'],
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
