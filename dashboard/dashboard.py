import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
import os

sns.set(style='dark')

# ─────────────────────────────────────────────
# Custom CSS – fix date picker width
# ─────────────────────────────────────────────

st.markdown("""
    <style>
        /* Kontainer date input */
        div[data-testid="stDateInput"] {
            width: 100% !important;
        }

        /* Input field */
        div[data-testid="stDateInput"] input {
            font-size: 12px !important;
            padding: 4px 8px !important;
        }

        /* Kalender popup */
        div[data-baseweb="calendar"] {
            transform: scale(0.80);
            transform-origin: top left;
        }

        /* Popover/overlay kalender agar tidak terpotong */
        div[data-baseweb="popover"] {
            transform: scale(0.80);
            transform-origin: top left;
        }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

weather_labels = {1: 'Cerah', 2: 'Berkabut', 3: 'Hujan Ringan', 4: 'Cuaca Buruk'}

def group_time_of_day(hour):
    if 0 <= hour <= 5:
        return 'Dini Hari'
    elif 6 <= hour <= 9:
        return 'Pagi (Rush)'
    elif 10 <= hour <= 14:
        return 'Siang'
    elif 15 <= hour <= 18:
        return 'Sore (Rush)'
    else:
        return 'Malam'


def create_weather_avg_df(df):
    avg_df = df.groupby('weathersit')['cnt'].mean().reset_index()
    avg_df['weathersit_label'] = avg_df['weathersit'].map(weather_labels)
    return avg_df


def create_hourly_weather_df(df):
    df = df.copy()
    df['weathersit_label'] = df['weathersit'].map(weather_labels)
    return df.groupby(['hr', 'weathersit_label'])['cnt'].mean().unstack()


def create_scatter_df(df):
    daily = df.groupby('dteday').agg(
        temp_actual=('temp_actual', 'mean'),
        hum_actual=('hum_actual', 'mean'),
        windspeed_actual=('windspeed_actual', 'mean'),
        cnt=('day_cnt', 'first')
    ).reset_index()
    return daily


def create_workingday_user_df(df):
    daily = df.groupby(['dteday', 'workingday']).agg(
        casual=('casual', 'sum'),
        registered=('registered', 'sum')
    ).reset_index()
    return daily.groupby('workingday')[['casual', 'registered']].mean()


def create_hourly_workingday_df(df):
    return df.groupby(['hr', 'workingday'])['cnt'].mean().unstack()


def create_hourly_user_type_df(df):
    work = df[df['workingday'] == 1].groupby('hr')[['casual', 'registered']].mean()
    holi = df[df['workingday'] == 0].groupby('hr')[['casual', 'registered']].mean()
    return work, holi


def create_time_segment_df(df):
    df = df.copy()
    df['time_of_day'] = df['hr'].apply(group_time_of_day)
    order = ['Dini Hari', 'Pagi (Rush)', 'Siang', 'Sore (Rush)', 'Malam']
    avg_cnt   = df.groupby('time_of_day')['cnt'].mean().reindex(order)
    total_cnt = df.groupby('time_of_day')['cnt'].sum().reindex(order)
    hour_avg  = df.groupby('hr')['cnt'].mean()
    return avg_cnt, total_cnt, hour_avg


# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    # Resolve path: support both local run and deployed environment
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'main_data.csv')

    main_data = pd.read_csv(csv_path)
    main_data['dteday'] = pd.to_datetime(main_data['dteday'])

    cat_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
    main_data[cat_cols] = main_data[cat_cols].astype('category')

    main_data.sort_values(by=['dteday', 'hr'], inplace=True)
    main_data.reset_index(drop=True, inplace=True)

    return main_data


main_data = load_data()

# ─────────────────────────────────────────────
# Sidebar – Filter Tanggal
# ─────────────────────────────────────────────

min_date = main_data['dteday'].min()
max_date = main_data['dteday'].max()

with st.sidebar:
    st.title("🚲 Bike Sharing Dashboard")
    st.markdown("---")

    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date],
    )

    st.markdown("---")

# ─────────────────────────────────────────────
# Filter Data
# ─────────────────────────────────────────────

filtered_data = main_data[
    (main_data['dteday'] >= str(start_date)) &
    (main_data['dteday'] <= str(end_date))
]

# ─────────────────────────────────────────────
# Header & Metric Cards
# ─────────────────────────────────────────────

st.title('Dashboard Projek Analisa Data')
st.markdown('Analisis penyewaan sepeda berdasarkan kondisi cuaca, tipe pengguna, dan pola waktu.')
st.markdown('---')

daily_summary = filtered_data.groupby('dteday').agg(
    cnt=('day_cnt', 'first'),
    casual=('casual', 'sum'),
    registered=('registered', 'sum')
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Penyewaan", f"{int(daily_summary['cnt'].sum()):,}")
with col2:
    st.metric("Rata-rata Harian", f"{int(daily_summary['cnt'].mean()):,}")
with col3:
    st.metric("Total Registered", f"{int(daily_summary['registered'].sum()):,}")
with col4:
    st.metric("Total Casual", f"{int(daily_summary['casual'].sum()):,}")

st.markdown('---')

# ─────────────────────────────────────────────
# Pertanyaan Bisnis 1 – Pengaruh Cuaca
# ─────────────────────────────────────────────

st.header('Pertanyaan 1: Apakah Kondisi Cuaca Mempengaruhi Penyewaan Sepeda?')

# Plot : Bar chart rata-rata per kondisi cuaca
weather_avg = create_weather_avg_df(filtered_data)
order_weather = ['Cerah', 'Berkabut', 'Hujan Ringan', 'Cuaca Buruk']
weather_avg['weathersit_label'] = pd.Categorical(
    weather_avg['weathersit_label'], categories=order_weather, ordered=True
)
weather_avg = weather_avg.sort_values('weathersit_label')

fig1, ax1 = plt.subplots(figsize=(8, 4))
bars = ax1.bar(weather_avg['weathersit_label'], weather_avg['cnt'], color='#2878B5', edgecolor='white', linewidth=1.2)
ax1.set_title('Rata-rata Penyewaan per Kondisi Cuaca (per Jam)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Kondisi Cuaca')
ax1.set_ylabel('Rata-rata Penyewaan')
for bar, val in zip(bars, weather_avg['cnt']):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig1)

st.info(
    "**Insight:** Cuaca cerah menghasilkan rata-rata penyewaan tertinggi (205/jam), "
    "diikuti berkabut, hujan ringan, dan cuaca buruk. Semakin buruk cuaca, semakin sedikit penyewa."
)

# Plot : Line chart per jam berdasarkan kondisi cuaca
hourly_weather = create_hourly_weather_df(filtered_data)

fig2, ax2 = plt.subplots(figsize=(12, 5))
color_map = {'Cerah': '#2ecc71', 'Berkabut': '#f39c12',
             'Hujan Ringan': '#3498db', 'Cuaca Buruk': '#e74c3c'}
for col in hourly_weather.columns:
    ax2.plot(hourly_weather.index, hourly_weather[col],
             marker='o', markersize=3, label=col, color=color_map.get(col))
ax2.set_title('Rata-rata Penyewaan tiap Jam berdasarkan Kondisi Cuaca', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Jam')
ax2.set_ylabel('Rata-rata Penyewaan')
ax2.set_xticks(range(0, 24))
ax2.legend(title='Kondisi Cuaca', bbox_to_anchor=(1.01, 1), loc='upper left')
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig2)

# Plot : Scatter plot variabel cuaca vs penyewaan
scatter_df = create_scatter_df(filtered_data)

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
axes3[0].scatter(scatter_df['temp_actual'], scatter_df['cnt'], alpha=0.5, color='#2878B5')
axes3[0].set_title('Suhu vs Penyewaan')
axes3[0].set_xlabel('Suhu (°C)')
axes3[0].set_ylabel('Jumlah Penyewaan')
axes3[0].spines[['top', 'right']].set_visible(False)

axes3[1].scatter(scatter_df['hum_actual'], scatter_df['cnt'], alpha=0.5, color='#2878B5')
axes3[1].set_title('Kelembaban vs Penyewaan')
axes3[1].set_xlabel('Kelembaban (%)')
axes3[1].set_ylabel('Jumlah Penyewaan')
axes3[1].spines[['top', 'right']].set_visible(False)

axes3[2].scatter(scatter_df['windspeed_actual'], scatter_df['cnt'], alpha=0.5, color='#2878B5')
axes3[2].set_title('Kecepatan Angin vs Penyewaan')
axes3[2].set_xlabel('Kecepatan Angin (km/h)')
axes3[2].set_ylabel('Jumlah Penyewaan')
axes3[2].spines[['top', 'right']].set_visible(False)

plt.suptitle('Hubungan Variabel Cuaca dengan Jumlah Penyewaan', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
st.pyplot(fig3)

st.info(
    "**Insight:** Suhu berkorelasi positif dengan penyewaan, semakin hangat semakin banyak penyewa. "
    "Kecepatan angin menunjukkan korelasi negatif lemah."
)

st.markdown('---')

# ─────────────────────────────────────────────
# Pertanyaan Bisnis 2 – Hari Kerja vs Libur
# ─────────────────────────────────────────────

st.header('Pertanyaan 2: Perbedaan Tren Hari Kerja vs Libur dan Registered vs Casual?')

# Plot : Bar chart rata-rata casual & registered
workday_user = create_workingday_user_df(filtered_data)

fig4, ax4 = plt.subplots(figsize=(7, 4))
x = np.arange(2)
width = 0.35
bars_cas = ax4.bar(x - width/2, workday_user['casual'],     width, label='Casual',     color='#f39c12', edgecolor='white')
bars_reg = ax4.bar(x + width/2, workday_user['registered'], width, label='Registered', color='#2980b9', edgecolor='white')
ax4.set_xticks(x)
ax4.set_xticklabels(['Hari Libur', 'Hari Kerja'])
ax4.set_title('Rata-rata Pengguna Casual & Registered\nHari Kerja vs Hari Libur', fontsize=13, fontweight='bold', pad=12)
ax4.set_ylabel('Rata-rata Penyewaan')
ax4.legend()
ax4.spines[['top', 'right']].set_visible(False)
for bar in list(bars_cas) + list(bars_reg):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
st.pyplot(fig4)

# Plot : Line chart penyewaan per jam hari kerja vs libur
hourly_wd = create_hourly_workingday_df(filtered_data)

fig5, ax5 = plt.subplots(figsize=(12, 5))
ax5.plot(hourly_wd.index, hourly_wd[0], marker='o', markersize=4, label='Hari Libur', color='#f39c12', linewidth=2)
ax5.plot(hourly_wd.index, hourly_wd[1], marker='o', markersize=4, label='Hari Kerja', color='#2980b9', linewidth=2)
ax5.set_title('Rata-rata Penyewaan tiap Jam: Hari Kerja vs Hari Libur', fontsize=13, fontweight='bold', pad=12)
ax5.set_xlabel('Jam')
ax5.set_ylabel('Rata-rata Penyewaan')
ax5.set_xticks(range(0, 24))
ax5.legend(title='Jenis Hari')
ax5.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig5)

st.info(
    "**Insight:** Hari kerja menunjukkan 2 puncak jelas di jam 8 pagi & 5 sore (jam komuter). "
    "Hari libur memiliki pola landai memanjang dari pagi hingga sore."
)

# Plot : Perbandingan casual vs registered per jam
work_df, holi_df = create_hourly_user_type_df(filtered_data)

fig6, axes6 = plt.subplots(1, 2, figsize=(15, 5))
axes6[0].plot(work_df.index, work_df['casual'],     marker='o', markersize=3, label='Casual',     color='#f39c12', linewidth=2)
axes6[0].plot(work_df.index, work_df['registered'], marker='o', markersize=3, label='Registered', color='#2980b9', linewidth=2)
axes6[0].set_title('Rata-rata per Jam – Hari Kerja', fontsize=12, fontweight='bold')
axes6[0].set_xlabel('Jam')
axes6[0].set_ylabel('Rata-rata Penyewaan')
axes6[0].set_xticks(range(0, 24))
axes6[0].legend()
axes6[0].spines[['top', 'right']].set_visible(False)

axes6[1].plot(holi_df.index, holi_df['casual'],     marker='o', markersize=3, label='Casual',     color='#f39c12', linewidth=2)
axes6[1].plot(holi_df.index, holi_df['registered'], marker='o', markersize=3, label='Registered', color='#2980b9', linewidth=2)
axes6[1].set_title('Rata-rata per Jam – Hari Libur', fontsize=12, fontweight='bold')
axes6[1].set_xlabel('Jam')
axes6[1].set_ylabel('Rata-rata Penyewaan')
axes6[1].set_xticks(range(0, 24))
axes6[1].legend()
axes6[1].spines[['top', 'right']].set_visible(False)

plt.suptitle('Perbandingan Pengguna Casual vs Registered per Jam', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
st.pyplot(fig6)

st.info(
    "**Insight:** Hari kerja didominasi pengguna registered dengan pola rush-hour. "
    "Hari libur memperlihatkan peningkatan signifikan pengguna casual dan polanya lebih mirip registered."
)

st.markdown('---')

# ─────────────────────────────────────────────
# Analisis Lanjutan – Segmentasi Waktu
# ─────────────────────────────────────────────

st.header('Analisis Lanjutan: Segmentasi Waktu Penyewaan')

avg_cnt, total_cnt, hour_avg = create_time_segment_df(filtered_data)

order      = ['Dini Hari', 'Pagi (Rush)', 'Siang', 'Sore (Rush)', 'Malam']
seg_colors = ['#6C5CE7', '#FDCB6E', '#00B894', '#E17055', '#74B9FF']

fig7, axes7 = plt.subplots(1, 2, figsize=(14, 6))

wedges, texts, autotexts = axes7[0].pie(
    total_cnt.values,
    labels=None,
    colors=seg_colors,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
    pctdistance=0.75
)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')
pie_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(seg_colors, order)]
axes7[0].legend(handles=pie_patches, loc='lower center',
                bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
axes7[0].set_title('Persentase Penyewaan per Segmen Waktu', fontweight='bold', pad=15)

hour_colors_map = {h: seg_colors[order.index(group_time_of_day(h))] for h in range(24)}
bar_colors = [hour_colors_map[h] for h in range(24)]
axes7[1].bar(range(24), hour_avg.values, color=bar_colors, edgecolor='white')
axes7[1].set_xticks(range(24))
axes7[1].set_xticklabels(range(24))
axes7[1].set_title('Rata-rata Penyewaan per Jam', fontweight='bold', pad=15)
axes7[1].set_xlabel('Jam')
axes7[1].set_ylabel('Rata-rata Penyewaan')
axes7[1].spines[['top', 'right']].set_visible(False)
legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(seg_colors, order)]
axes7[1].legend(handles=legend_patches, loc='upper left', fontsize=9)

plt.suptitle('Manual Grouping: Segmentasi Waktu Penyewaan', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
st.pyplot(fig7)

st.info(
    "**Insight:** Sore merupakan waktu dengan jumlah penyewa sepeda terbanyak. Keumdian disusul oleh siang dan malam "
    "Jika dilihat dari frekuensi per jamnya, waktu paling sibuk terjadi pada pagi dan sore hari, terbukti dengan tiga frekuensi tertinggi terjadi pada jam 8, 17, dan 18."
)

st.markdown('---')