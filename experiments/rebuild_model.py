import pandas as pd, numpy as np, pickle, sys, os, warnings
warnings.filterwarnings('ignore')
import geopandas as gpd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
from pathlib import Path
from branca.colormap import LinearColormap
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
sys.stdout.reconfigure(encoding='utf-8')

DATA  = Path(r'C:\Users\ibarn\Documents\GitHub\ml-jenki-forecasting-models\data\source-data')
OUT   = Path(r'C:\Users\ibarn\Documents\GitHub\ml-jenki-forecasting-models\experiments')
PLOTS = Path(r'C:\Users\ibarn\Documents\GitHub\ml-jenki-forecasting-models\plots')
SHP   = Path(r'C:\Users\ibarn\AppData\Local\Temp\oa_shapes\OA_2021_EW_BFE_V9.shp')

# 1. Training data
persona = pd.read_csv(r'C:\Users\ibarn\Downloads\persona_oa_props_impactScore_comm.csv').set_index('code')
uk_avg = persona.mean()

STORE_OAS      = {'borough':'E00019779','battersea':'E00183178','canary_wharf':'E00167122',
                  'covent_garden':'E00004529','spitalfields':'E00021686'}
STORE_REVENUE  = {'borough':2329.35,'battersea':2410.99,'canary_wharf':1961.93,
                  'covent_garden':2254.21,'spitalfields':3751.40}
STORE_POP      = {'borough':331,'battersea':289,'canary_wharf':458,'covent_garden':278,'spitalfields':644}

rows = []
for store, oa in STORE_OAS.items():
    rows.append({
        'store': store,
        'mean_daily_revenue': STORE_REVENUE[store],
        'bombe_6_delta': persona.loc[oa,'Bombe 6'] - uk_avg['Bombe 6'],
        'bombe_7_delta': persona.loc[oa,'Bombe 7'] - uk_avg['Bombe 7'],
        'population':    STORE_POP[store],
    })

train_df = pd.DataFrame(rows).set_index('store')
print('Training data:')
print(train_df.to_string())

feature_cols = ['bombe_6_delta', 'bombe_7_delta', 'population']
X = train_df[feature_cols].values.astype(float)
y = train_df['mean_daily_revenue'].values.astype(float)

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

ridge = RidgeCV(alphas=np.logspace(-1, 5, 100), cv=LeaveOneOut())
ridge.fit(X_sc, y)
print(f'\nBest alpha: {ridge.alpha_:.2f}')

loo_preds = np.zeros(len(y))
for train_idx, test_idx in LeaveOneOut().split(X_sc):
    m = Ridge(alpha=ridge.alpha_)
    m.fit(X_sc[train_idx], y[train_idx])
    loo_preds[test_idx] = m.predict(X_sc[test_idx])

mae = np.mean(np.abs(loo_preds - y))
print(f'LOOCV MAE: {mae:.0f} ({mae/y.mean()*100:.1f}% of mean)')
for i, store in enumerate(train_df.index):
    print(f'  {store:15s}  actual={y[i]:.0f}  pred={loo_preds[i]:.0f}  err={y[i]-loo_preds[i]:+.0f}')

pickle.dump(ridge,  open(OUT / 'ridge_model_v2.pkl','wb'))
pickle.dump(scaler, open(OUT / 'ridge_scaler_v2.pkl','wb'))

# 2. Feature importance plot
labels = ['Bombe 6 Delta', 'Bombe 7 Delta', 'Population']
coef_df = pd.DataFrame({'label': labels, 'coef': ridge.coef_}).sort_values('coef')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
colors = ['#2166ac' if v > 0 else '#d6604d' for v in coef_df['coef']]
bars = ax.barh(coef_df['label'], coef_df['coef'], color=colors, edgecolor='white', height=0.5)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Ridge Coefficient (scaled units)', fontsize=10)
ax.set_title('Feature Importance\n(positive = drives revenue up)', fontweight='bold')
for bar, val in zip(bars, coef_df['coef']):
    ax.text(val + (5 if val >= 0 else -5), bar.get_y() + bar.get_height()/2,
            f'{val:+,.0f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
ax.set_xlim(coef_df['coef'].min()*1.4, coef_df['coef'].max()*1.4)

store_colors = {'borough':'#4e79a7','battersea':'#f28e2b','canary_wharf':'#e15759',
                'covent_garden':'#76b7b2','spitalfields':'#59a14f'}
ax2 = axes[1]
for i, store in enumerate(train_df.index):
    ax2.scatter(y[i], loo_preds[i], label=store.replace('_',' ').title(),
                color=store_colors[store], s=100, zorder=3)
lims = [min(y.min(),loo_preds.min())*0.88, max(y.max(),loo_preds.max())*1.05]
ax2.plot(lims, lims, 'k--', linewidth=1, label='Perfect fit')
ax2.set_xlabel('Actual Mean Daily Revenue', fontsize=10)
ax2.set_ylabel('LOOCV Predicted', fontsize=10)
ax2.set_title(f'Actual vs LOOCV Predicted\nMAE=£{mae:.0f}', fontweight='bold')
ax2.legend(fontsize=8); ax2.set_xlim(lims); ax2.set_ylim(lims)
plt.tight_layout()
plt.savefig(PLOTS / 'feature_importance.png', dpi=150, bbox_inches='tight')
print('\nFeature importance plot saved.')

# 3. Score all London OAs
print('Scoring all London OAs...')
oa_rgn = pd.read_csv(DATA / 'OA21_RGN22_LU.csv')
london_oas = set(oa_rgn[oa_rgn['rgn22nm'] == 'London']['oa21cd'].unique())
london_persona = persona[persona.index.isin(london_oas)].copy()

sape = pd.read_excel(
    os.path.join(os.environ['TEMP'], 'sape.xlsx'),
    sheet_name='Mid-2022 OA 2021', header=3, usecols=['OA 2021 Code','Total']
)
sape.columns = ['oa21cd','population']
sape = sape.dropna(subset=['oa21cd'])
sape['population'] = pd.to_numeric(sape['population'], errors='coerce')
sape_idx = sape.set_index('oa21cd')

scores = london_persona[['Bombe 6','Bombe 7']].copy()
scores['bombe_6_delta'] = scores['Bombe 6'] - uk_avg['Bombe 6']
scores['bombe_7_delta'] = scores['Bombe 7'] - uk_avg['Bombe 7']
scores['population'] = sape_idx.reindex(scores.index)['population'].fillna(sape_idx['population'].median())
scores = scores[feature_cols].dropna()

X_all_sc = scaler.transform(scores.values.astype(float))
scores['predicted_revenue'] = ridge.predict(X_all_sc)
scores['predicted_revenue'] = scores['predicted_revenue'].clip(
    lower=300.0, upper=scores['predicted_revenue'].quantile(0.99)
)
print(f'Scored {len(scores):,} OAs')
print(scores['predicted_revenue'].describe().round(0))

# 4. Load London-only shapefile
print('Loading shapefile (London only)...')
gdf = gpd.read_file(SHP)
gdf_london = gdf[gdf['OA21CD'].isin(london_oas)].copy()
gdf_london = gdf_london.to_crs('EPSG:4326')
gdf_london['geometry'] = gdf_london['geometry'].simplify(tolerance=0.0001, preserve_topology=True)
gdf_london = gdf_london.merge(
    scores[['predicted_revenue']].reset_index().rename(columns={'code':'OA21CD'}),
    on='OA21CD', how='left'
)
med = scores['predicted_revenue'].median()
gdf_london['predicted_revenue'] = gdf_london['predicted_revenue'].fillna(med)
print(f'London OAs in map: {len(gdf_london):,}')

# 5. Folium map
print('Rendering map...')
m = folium.Map(location=[51.505, -0.09], zoom_start=10,
               tiles='CartoDB positron', prefer_canvas=True)

vmin = scores['predicted_revenue'].quantile(0.05)
vmax = scores['predicted_revenue'].quantile(0.95)
colormap = LinearColormap(
    colors=['#deebf7','#9ecae1','#4292c6','#2171b5','#084594'],
    vmin=vmin, vmax=vmax, caption='Predicted Mean Daily Revenue (£)'
)

folium.GeoJson(
    gdf_london[['geometry','OA21CD','predicted_revenue']].to_json(),
    style_function=lambda feat: {
        'fillColor':   colormap(float(feat['properties']['predicted_revenue'] or vmin)),
        'color':       'none',
        'fillOpacity': 0.75,
        'weight':      0,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['OA21CD','predicted_revenue'],
        aliases=['OA Code','Est. Daily Revenue (£)'],
        localize=True,
    ),
).add_to(m)
colormap.add_to(m)

STORE_LATLNG = {
    'Borough':(51.5055,-0.0908),'Battersea':(51.4839,-0.1445),
    'Canary Wharf':(51.5054,-0.0235),'Covent Garden':(51.5129,-0.1243),
    'Spitalfields':(51.5194,-0.0749),
}
STORE_REV_ACTUAL = {
    'Borough':2329,'Battersea':2411,'Canary Wharf':1962,'Covent Garden':2254,'Spitalfields':3751
}
for name, (lat, lon) in STORE_LATLNG.items():
    folium.CircleMarker(
        location=[lat, lon], radius=8,
        color='#ff4444', fill=True, fill_color='#ff4444', fill_opacity=0.9,
        tooltip=f'<b>{name}</b><br>Actual avg: £{STORE_REV_ACTUAL[name]:,}/day',
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
