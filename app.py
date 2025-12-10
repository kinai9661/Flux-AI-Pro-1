import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import datetime
import base64
from typing import Dict, List, Tuple
import time
import random
import json
import uuid
import os
import re
from urllib.parse import urlencode, quote
import gc
from streamlit.errors import StreamlitAPIException, StreamlitSecretNotFoundError

# 為免費方案設定限制
MAX_HISTORY_ITEMS = 15
MAX_FAVORITE_ITEMS = 30
MAX_BATCH_SIZE = 4

# 圖像尺寸預設
IMAGE_SIZES = {
    "自定義...": "Custom", "1024x1024": "正方形 (1:1)", "1080x1080": "IG 貼文 (1:1)",
    "1080x1350": "IG 縱向 (4:5)", "1080x1920": "IG Story (9:16)", "1200x630": "FB 橫向 (1.91:1)",
}

# 風格預設
STYLE_PRESETS = {
    "無": "", "電影感": "cinematic, dramatic lighting, high detail, sharp focus, epic",
    "動漫風": "anime, manga style, vibrant colors, clean line art, studio ghibli", 
    "賽博龐克": "cyberpunk, neon lights, futuristic city, high-tech, Blade Runner",
    "印象派": "impressionism, soft light, visible brushstrokes, Monet style", 
    "超現實主義": "surrealism, dreamlike, bizarre, Salvador Dali style",
    "普普藝術": "pop art, bold colors, comic book style, Andy Warhol", 
    "水墨畫": "ink wash painting, traditional chinese art, minimalist, zen",
    "3D 模型": "3d model, octane render, unreal engine, hyperdetailed, 4k", 
    "像素藝術": "pixel art, 16-bit, retro gaming style, sprite sheet",
    "低面建模": "low poly, simple shapes, vibrant color palette, isometric", 
    "矢量圖": "vector art, flat design, clean lines, graphic illustration",
    "蒸汽龐克": "steampunk, victorian, gears, clockwork, intricate details", 
    "黑暗奇幻": "dark fantasy, gothic, grim, lovecraftian horror, moody lighting",
    "水彩畫": "watercolor painting, soft wash, blended colors, delicate", 
    "剪紙藝術": "paper cut-out, layered paper, papercraft, flat shapes",
    "奇幻藝術": "fantasy art, epic, detailed, magical, lord of the rings", 
    "漫畫書": "comic book art, halftone dots, bold outlines, graphic novel style",
    "線條藝術": "line art, monochrome, minimalist, clean lines", 
    "霓虹龐克": "neon punk, fluorescent, glowing, psychedelic, vibrant",
    "黑白線條藝術": "black and white line art, minimalist, clean vector, coloring book style",
}

# === 黑金主題系統 ===
def inject_black_gold_theme():
    st.markdown("""
    <style>
    /* Google Fonts 導入 */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600;700&family=Noto+Sans+TC:wght@400;500;700;900&display=swap');
    
    :root {
        --primary-gold: #FFD700;
        --secondary-gold: #FFA500;
        --dark-gold: #FF8C00;
        --light-gold: #F5DEB3;
        --pure-black: #000000;
        --deep-black: #0a0a0a;
        --card-black: #1a1a1a;
        --border-gold: rgba(255, 215, 0, 0.3);
        --gold-gradient: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FF8C00 100%);
        --gold-glow: 0 0 20px rgba(255, 215, 0, 0.5);
        --gold-shadow: 0 8px 32px rgba(255, 215, 0, 0.3);
        --font-heading: 'Playfair Display', 'Noto Sans TC', serif;
        --font-body: 'Inter', 'Noto Sans TC', sans-serif;
    }
    
    * { font-family: var(--font-body); }
    
    .main {
        background: radial-gradient(ellipse at top, #1a1a1a 0%, #000000 100%);
        background-attachment: fixed;
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    h1, h2, h3 {
        font-family: var(--font-heading) !important;
        font-weight: 900 !important;
        letter-spacing: 0.5px;
    }
    
    h1 {
        background: var(--gold-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem !important;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: var(--primary-gold);
        font-size: 1.8rem !important;
        border-bottom: 2px solid var(--border-gold);
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: var(--light-gold);
        font-size: 1.4rem !important;
    }
    
    .stButton > button {
        background: var(--gold-gradient);
        color: var(--pure-black);
        border: 2px solid var(--primary-gold);
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--gold-shadow);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.05);
        box-shadow: 0 16px 48px rgba(255, 215, 0, 0.6);
        border-color: var(--light-gold);
    }
    
    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, var(--card-black) 0%, var(--deep-black) 100%);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 2px solid var(--border-gold);
        transition: all 0.4s ease;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    }
    
    div[data-testid="stExpander"]:hover {
        border-color: var(--primary-gold);
        box-shadow: var(--gold-shadow);
        transform: translateY(-4px);
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background: var(--deep-black) !important;
        border: 2px solid var(--border-gold) !important;
        border-radius: 12px !important;
        color: var(--light-gold) !重要;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div:focus-within {
        border-color: var(--primary-gold) !important;
        box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.2) !important;
        background: var(--card-black) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: var(--deep-black);
        padding: 12px;
        border-radius: 16px;
        border: 1px solid var(--border-gold);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 14px 28px;
        color: var(--light-gold);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gold-gradient) !important;
        color: var(--pure-black) !important;
        box-shadow: var(--gold-shadow);
    }
    
    .element-container img {
        border-radius: 16px;
        border: 2px solid var(--border-gold);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .element-container img:hover {
        transform: scale(1.05) translateY(-8px);
        border-color: var(--primary-gold);
        box-shadow: var(--gold-shadow), 0 16px 48px rgba(0, 0, 0, 0.8);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--pure-black) 0%, var(--deep-black) 100%);
        border-right: 2px solid var(--border-gold);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 165, 0, 0.05) 100%);
        border-left: 4px solid var(--primary-gold);
        border-radius: 12px;
        padding: 1.25rem;
        color: var(--light-gold);
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.1) 0%, rgba(255, 140, 0, 0.05) 100%);
        border-left: 4px solid var(--secondary-gold);
        border-radius: 12px;
        padding: 1.25rem;
        color: var(--light-gold);
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.1) 0%, rgba(185, 28, 28, 0.05) 100%);
        border-left: 4px solid #dc2626;
        border-radius: 12px;
        padding: 1.25rem;
        color: #fca5a5;
    }
    
    .stProgress > div > div > div {
        background: var(--gold-gradient) !important;
        border-radius: 10px;
        box-shadow: var(--gold-glow);
    }
    
    .stProgress > div > div {
        background: var(--deep-black) !important;
        border: 1px solid var(--border-gold);
    }
    
    .main p, .main li, .main span {
        color: var(--light-gold);
        line-height: 1.7;
    }
    
    .main strong, .main b {
        color: var(--primary-gold);
        font-weight: 700;
    }
    
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--deep-black);
        border-radius: 10px;
        border: 1px solid var(--border-gold);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gold-gradient);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #FFE55C 0%, #FFB347 100%);
        box-shadow: var(--gold-glow);
    }
    
    @media (max-width: 768px) {
        .main .block-container { padding: 1rem; }
        h1 { font-size: 2rem !important; }
        .stButton > button { width: 100%; margin: 0.5rem 0; }
    }
    </style>
    """, unsafe_allow_html=True)

# 其餘業務邏輯保持不變（此處省略重複代碼）
