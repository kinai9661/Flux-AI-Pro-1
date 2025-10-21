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
MAX_HISTORY_ITEMS = 20
MAX_FAVORITE_ITEMS = 40
MAX_BATCH_SIZE = 4

# 圖像尺寸預設
IMAGE_SIZES = {
    "自定義...": "Custom", 
    "512x512": "SD 標準 (1:1)", 
    "768x768": "SD XL 標準 (1:1)",
    "1024x1024": "正方形 (1:1)", 
    "1080x1080": "IG 貼文 (1:1)",
    "512x768": "SD 縱向 (2:3)",
    "768x512": "SD 橫向 (3:2)",
    "1080x1350": "IG 縱向 (4:5)", 
    "1080x1920": "IG Story (9:16)", 
    "1200x630": "FB 橫向 (1.91:1)",
    "1536x640": "超寬橫幅 (2.4:1)",
    "896x1152": "肖像模式 (7:9)",
    "1152x896": "風景模式 (9:7)",
}

# 擴展風格預設
STYLE_PRESETS = {
    # 基礎風格
    "無": "", 
    "電影感": "cinematic, dramatic lighting, high detail, sharp focus, epic, movie still",
    "動漫風": "anime, manga style, vibrant colors, clean line art, studio ghibli style", 
    "賽博龐克": "cyberpunk, neon lights, futuristic city, high-tech, Blade Runner style",
    # 藝術流派
    "印象派": "impressionism, soft light, visible brushstrokes, Monet style, oil painting", 
    "超現實主義": "surrealism, dreamlike, bizarre, Salvador Dali style, melting clocks",
    "普普藝術": "pop art, bold colors, comic book style, Andy Warhol, Roy Lichtenstein", 
    "水墨畫": "ink wash painting, traditional chinese art, minimalist, zen, black ink on white paper",
    # 數位與遊戲風格
    "3D 模型": "3d model, octane render, unreal engine 5, hyperdetailed, 4k, volumetric lighting", 
    "像素藝術": "pixel art, 16-bit, retro gaming style, sprite sheet, pixelated",
    "低面建模": "low poly, simple shapes, vibrant color palette, isometric, geometric", 
    "矢量圖": "vector art, flat design, clean lines, graphic illustration, adobe illustrator style",
    # 幻想與特定風格
    "蒸汽龐克": "steampunk, victorian era, brass gears, clockwork, copper pipes, intricate details", 
    "黑暗奇幻": "dark fantasy, gothic, grim, lovecraftian horror, moody lighting, dark atmosphere",
    "水彩畫": "watercolor painting, soft wash, blended colors, delicate, paper texture", 
    "剪紙藝術": "paper cut-out, layered paper, papercraft, flat shapes, shadow box",
    "奇幻藝術": "fantasy art, epic, detailed, magical, lord of the rings style, dragons and wizards", 
    "漫畫書": "comic book art, halftone dots, bold outlines, graphic novel style, superhero",
    "線條藝術": "line art, monochrome, minimalist, clean lines, pen and ink", 
    "霓虹龐克": "neon punk, fluorescent, glowing, psychedelic, vibrant neon colors",
    "黑白線條藝術": "black and white line art, minimalist, clean vector, coloring book style",
    # 新增攝影風格
    "人像攝影": "portrait photography, professional headshot, studio lighting, bokeh background",
    "街頭攝影": "street photography, candid, urban, documentary style, natural lighting",
    "風景攝影": "landscape photography, golden hour, wide angle, nature, scenic vista",
    "微距攝影": "macro photography, extreme close-up, detailed textures, shallow depth of field",
    # 新增藝術風格
    "抽象表現主義": "abstract expressionism, Jackson Pollock style, paint splatters, emotional",
    "立體主義": "cubism, Pablo Picasso style, geometric shapes, fragmented perspective",
    "新藝術運動": "art nouveau, ornate decorations, flowing lines, Alphonse Mucha style",
    "包豪斯": "bauhaus style, geometric, functional design, minimalist, clean typography",
    "復古海報": "vintage poster, retro advertising, pin-up style, 1950s aesthetic",
}

def rerun_app():
    if hasattr(st, 'rerun'): st.rerun()
    elif hasattr(st, 'experimental_rerun'): st.experimental_rerun()
    else: st.stop()

st.set_page_config(page_title="AI 圖像生成器 (多模型版)", page_icon="🎨", layout="wide")

# 大幅擴展的API供應商和模型配置
API_PROVIDERS = {
    "Pollinations.ai": {
        "name": "Pollinations.ai Studio", 
        "base_url_default": "https://image.pollinations.ai", 
        "icon": "🌸",
        "hardcoded_models": {
            # FLUX 系列
            "flux-1.1-pro": {"name": "Flux 1.1 Pro", "icon": "🏆", "category": "FLUX"},
            "flux.1-kontext-pro": {"name": "Flux.1 Kontext Pro", "icon": "🧠", "category": "FLUX"},
            "flux.1-kontext-max": {"name": "Flux.1 Kontext Max", "icon": "👑", "category": "FLUX"},
            "flux-dev": {"name": "Flux Dev", "icon": "🛠️", "category": "FLUX"},
            "flux-schnell": {"name": "Flux Schnell", "icon": "⚡", "category": "FLUX"},
            "flux-realism": {"name": "Flux Realism", "icon": "📷", "category": "FLUX"},
            # Stable Diffusion 系列
            "stable-diffusion-3.5-large": {"name": "SD 3.5 Large", "icon": "🎯", "category": "Stable Diffusion"},
            "stable-diffusion-3.5-medium": {"name": "SD 3.5 Medium", "icon": "⚖️", "category": "Stable Diffusion"},
            "stable-diffusion-xl": {"name": "SDXL 1.0", "icon": "💎", "category": "Stable Diffusion"},
            "stable-diffusion-xl-turbo": {"name": "SDXL Turbo", "icon": "🚀", "category": "Stable Diffusion"},
            "stable-diffusion-2.1": {"name": "SD 2.1", "icon": "🔄", "category": "Stable Diffusion"},
            "stable-diffusion-1.5": {"name": "SD 1.5", "icon": "🔰", "category": "Stable Diffusion"},
            # 專業模型
            "midjourney": {"name": "Midjourney", "icon": "🎭", "category": "Professional"},
            "dalle-3": {"name": "DALL-E 3", "icon": "🤖", "category": "OpenAI"},
            "playground-v2.5": {"name": "Playground v2.5", "icon": "🎪", "category": "Professional"},
            # 特化模型
            "dreamshaper": {"name": "DreamShaper", "icon": "💫", "category": "Community"},
            "realistic-vision": {"name": "Realistic Vision", "icon": "👁️", "category": "Community"},
            "deliberate": {"name": "Deliberate", "icon": "🎨", "category": "Community"},
            "anything-v5": {"name": "Anything v5", "icon": "🌟", "category": "Anime"},
            "waifu-diffusion": {"name": "Waifu Diffusion", "icon": "👩‍🎨", "category": "Anime"},
            "openjourney": {"name": "OpenJourney", "icon": "🗺️", "category": "Community"},
            # 風格特化模型
            "analog-diffusion": {"name": "Analog Film", "icon": "📸", "category": "Style"},
            "synthwave-diffusion": {"name": "Synthwave", "icon": "🌆", "category": "Style"},
            "cyberpunk-anime": {"name": "Cyberpunk Anime", "icon": "🤖", "category": "Style"},
            "pixel-art-xl": {"name": "Pixel Art XL", "icon": "🎮", "category": "Style"},
        }
    },
    "NavyAI": {
        "name": "NavyAI", 
        "base_url_default": "https://api.navy/v1", 
        "icon": "⚓",
        "hardcoded_models": {
            "flux-pro": {"name": "Flux Pro", "icon": "🏆", "category": "FLUX"},
            "flux-schnell": {"name": "Flux Schnell", "icon": "⚡", "category": "FLUX"},
            "stable-diffusion-xl": {"name": "SDXL", "icon": "💎", "category": "Stable Diffusion"},
            "midjourney-v6": {"name": "Midjourney v6", "icon": "🎭", "category": "Professional"},
        }
    },
    "OpenAI Compatible": {
        "name": "OpenAI 兼容 API", 
        "base_url_default": "https://api.openai.com/v1", 
        "icon": "🤖",
        "hardcoded_models": {
            "dall-e-3": {"name": "DALL-E 3", "icon": "🤖", "category": "OpenAI"},
            "dall-e-2": {"name": "DALL-E 2", "icon": "🔄", "category": "OpenAI"},
        }
    },
    "Hugging Face": {
        "name": "Hugging Face Inference",
        "base_url_default": "https://api-inference.huggingface.co",
        "icon": "🤗",
        "hardcoded_models": {
            "stable-diffusion-v1-5": {"name": "SD 1.5 (HF)", "icon": "🔰", "category": "Stable Diffusion"},
            "stable-diffusion-xl-base-1.0": {"name": "SDXL Base (HF)", "icon": "💎", "category": "Stable Diffusion"},
            "flux-1-dev": {"name": "Flux.1 Dev (HF)", "icon": "🛠️", "category": "FLUX"},
        }
    }
}

# 基礎模型集合
BASE_MODELS = {
    "flux.1-schnell": {"name": "FLUX.1 Schnell", "icon": "⚡", "priority": 1, "category": "FLUX"},
    "stable-diffusion-xl": {"name": "Stable Diffusion XL", "icon": "💎", "priority": 2, "category": "Stable Diffusion"},
    "stable-diffusion-1.5": {"name": "Stable Diffusion 1.5", "icon": "🔰", "priority": 3, "category": "Stable Diffusion"},
}

# --- 核心函數 ---
def init_session_state():
    if 'api_profiles' not in st.session_state:
        try: 
            base_profiles = st.secrets.get("api_profiles", {})
        except StreamlitSecretNotFoundError: 
            base_profiles = {}
        
        default_profiles = {
            "預設 Pollinations": {
                'provider': 'Pollinations.ai', 
                'api_key': '', 
                'base_url': 'https://image.pollinations.ai', 
                'validated': True, 
                'pollinations_auth_mode': '免費', 
                'pollinations_token': '', 
                'pollinations_referrer': ''
            }
        }
        
        st.session_state.api_profiles = base_profiles.copy() if base_profiles else default_profiles
        
    if 'active_profile_name' not in st.session_state or st.session_state.active_profile_name not in st.session_state.api_profiles:
        st.session_state.active_profile_name = list(st.session_state.api_profiles.keys())[0] if st.session_state.api_profiles else ""
    
    defaults = {
        'generation_history': [], 
        'favorite_images': [], 
        'discovered_models': {},
        'model_categories_expanded': {'FLUX': True, 'Stable Diffusion': True, 'Professional': False, 'Community': False, 'Anime': False, 'Style': False}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state: 
            st.session_state[key] = value

def get_active_config(): 
    return st.session_state.api_profiles.get(st.session_state.active_profile_name, {})

def auto_discover_models(client, provider, base_url) -> Dict[str, Dict]:
    discovered = {}
    try:
        if provider == "Pollinations.ai":
            response = requests.get(f"{base_url}/models", timeout=10)
            if response.ok:
                models = response.json()
                for model_name in models: 
                    # 智能分類
                    category = "Community"
                    if any(x in model_name.lower() for x in ['flux', 'kontext']):
                        category = "FLUX"
                    elif any(x in model_name.lower() for x in ['stable-diffusion', 'sd', 'sdxl']):
                        category = "Stable Diffusion"
                    elif any(x in model_name.lower() for x in ['anime', 'waifu', 'anything']):
                        category = "Anime"
                    elif any(x in model_name.lower() for x in ['midjourney', 'dalle', 'playground']):
                        category = "Professional"
                    
                    discovered[model_name] = {
                        "name": model_name.replace('-', ' ').replace('_', ' ').title(), 
                        "icon": "🌸",
                        "category": category
                    }
            else: 
                st.warning(f"無法從 Pollinations 獲取模型列表: HTTP {response.status_code}")
                
        elif client:
            models = client.models.list().data
            for model in models:
                if any(keyword in model.id.lower() for keyword in ['flux', 'stable', 'dall', 'midjourney', 'sd']):
                    # 智能分類和圖標
                    category = "Community"
                    icon = "🤖"
                    
                    if 'flux' in model.id.lower():
                        category = "FLUX"
                        icon = "⚡"
                    elif any(x in model.id.lower() for x in ['stable', 'sd']):
                        category = "Stable Diffusion"
                        icon = "💎"
                    elif 'dall' in model.id.lower():
                        category = "OpenAI"
                        icon = "🤖"
                    elif 'midjourney' in model.id.lower():
                        category = "Professional"
                        icon = "🎭"
                    
                    discovered[model.id] = {
                        "name": model.id.replace('-', ' ').replace('_', ' ').title(), 
                        "icon": icon,
                        "category": category
                    }
    except Exception as e: 
        st.error(f"發現模型失敗: {e}")
    return discovered

def merge_models() -> Dict[str, Dict]:
    provider = get_active_config().get('provider')
    discovered = st.session_state.get('discovered_models', {})
    
    if provider in API_PROVIDERS:
        hardcoded = API_PROVIDERS[provider].get('hardcoded_models', {})
        merged = {**hardcoded, **discovered}
    else:
        merged = {**BASE_MODELS, **discovered}
    
    return merged

def get_models_by_category(models: Dict[str, Dict]) -> Dict[str, Dict[str, Dict]]:
    """按類別組織模型"""
    categorized = {}
    for model_id, model_info in models.items():
        category = model_info.get('category', 'Other')
        if category not in categorized:
            categorized[category] = {}
        categorized[category][model_id] = model_info
    return categorized

def validate_api_key(api_key: str, base_url: str, provider: str) -> Tuple[bool, str]:
    if provider == "Pollinations.ai": 
        return True, "Pollinations.ai 無需驗證"
    elif provider == "Hugging Face":
        if not api_key:
            return False, "Hugging Face 需要 API Token"
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
            if response.status_code == 200:
                return True, "Hugging Face API Token 驗證成功"
            else:
                return False, f"Hugging Face API 驗證失敗: {response.status_code}"
        except Exception as e:
            return False, f"Hugging Face API 驗證失敗: {e}"
    else:
        try: 
            OpenAI(api_key=api_key, base_url=base_url).models.list()
            return True, "API 密鑰驗證成功"
        except Exception as e: 
            return False, f"API 驗證失敗: {e}"

def generate_images_with_retry(client, **params) -> Tuple[bool, any]:
    provider = get_active_config().get('provider')
    n_images = params.get("n", 1)
    
    if provider == "Pollinations.ai":
        return generate_pollinations_images(params, n_images)
    elif provider == "Hugging Face":
        return generate_huggingface_images(params, n_images)
    else:
        return generate_openai_compatible_images(client, params, n_images)

def generate_pollinations_images(params, n_images):
    generated_images = []
    cfg = get_active_config()
    
    for i in range(n_images):
        try:
            current_params = params.copy()
            current_params["seed"] = random.randint(0, 1000000)
            prompt = current_params.get("prompt", "")
            
            if (neg_prompt := current_params.get("negative_prompt")): 
                prompt += f" --no {neg_prompt}"
                
            width, height = str(current_params.get("size", "1024x1024")).split('x')
            
            api_params = {
                k: v for k, v in {
                    "model": current_params.get("model"), 
                    "width": width, 
                    "height": height, 
                    "seed": current_params.get("seed"), 
                    "nologo": current_params.get("nologo"), 
                    "private": current_params.get("private"), 
                    "enhance": current_params.get("enhance"), 
                    "safe": current_params.get("safe")
                }.items() if v is not None
            }
            
            headers = {}
            auth_mode = cfg.get('pollinations_auth_mode', '免費')
            
            if auth_mode == '令牌' and cfg.get('pollinations_token'): 
                headers['Authorization'] = f"Bearer {cfg['pollinations_token']}"
            elif auth_mode == '域名' and cfg.get('pollinations_referrer'): 
                headers['Referer'] = cfg['pollinations_referrer']
                
            url = f"{cfg['base_url']}/prompt/{quote(prompt)}?{urlencode(api_params)}"
            response = requests.get(url, headers=headers, timeout=120)
            
            if response.ok:
                b64_json = base64.b64encode(response.content).decode()
                image_obj = type('Image', (object,), {'b64_json': b64_json})
                generated_images.append(image_obj)
            else: 
                st.warning(f"第 {i+1} 張圖片生成失敗: HTTP {response.status_code}")
                
        except Exception as e:
            st.warning(f"第 {i+1} 張圖片生成時出錯: {e}")
            continue
            
    if generated_images:
        response_obj = type('Response', (object,), {'data': generated_images})
        return True, response_obj
    else: 
        return False, "所有圖片生成均失敗。"

def generate_huggingface_images(params, n_images):
    generated_images = []
    cfg = get_active_config()
    
    for i in range(n_images):
        try:
            headers = {"Authorization": f"Bearer {cfg['api_key']}"}
            model = params.get("model")
            prompt = params.get("prompt", "")
            
            # Hugging Face 推理 API 格式
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": params.get("negative_prompt", ""),
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5
                }
            }
            
            url = f"{cfg['base_url']}/models/{model}"
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.ok:
                b64_json = base64.b64encode(response.content).decode()
                image_obj = type('Image', (object,), {'b64_json': b64_json})
                generated_images.append(image_obj)
            else:
                st.warning(f"第 {i+1} 張圖片生成失敗: HTTP {response.status_code}")
                
        except Exception as e:
            st.warning(f"第 {i+1} 張圖片生成時出錯: {e}")
            continue
            
    if generated_images:
        response_obj = type('Response', (object,), {'data': generated_images})
        return True, response_obj
    else:
        return False, "所有圖片生成均失敗。"

def generate_openai_compatible_images(client, params, n_images):
    try:
        sdk_params = {
            "model": params.get("model"), 
            "prompt": params.get("prompt"), 
            "size": str(params.get("size")), 
            "n": n_images, 
            "response_format": "b64_json"
        }
        
        # 添加負向提示詞支持（如果API支持）
        if params.get("negative_prompt"):
            sdk_params["negative_prompt"] = params.get("negative_prompt")
            
        sdk_params = {k: v for k, v in sdk_params.items() if v is not None and v != ""}
        return True, client.images.generate(**sdk_params)
    except Exception as e: 
        return False, str(e)

def add_to_history(prompt: str, negative_prompt: str, model: str, images: List[str], metadata: Dict):
    history = st.session_state.generation_history
    history.insert(0, {
        "id": str(uuid.uuid4()), 
        "timestamp": datetime.datetime.now(), 
        "prompt": prompt, 
        "negative_prompt": negative_prompt, 
        "model": model, 
        "images": images, 
        "metadata": metadata
    })
    st.session_state.generation_history = history[:MAX_HISTORY_ITEMS]

def display_image_with_actions(b64_json: str, image_id: str, history_item: Dict):
    try:
        img_data = base64.b64decode(b64_json)
        st.image(Image.open(BytesIO(img_data)), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1: 
            st.download_button(
                "📥 下載", 
                img_data, 
                f"ai_generated_{image_id}.png", 
                "image/png", 
                key=f"dl_{image_id}", 
                use_container_width=True
            )
            
        with col2:
            is_fav = any(fav['id'] == image_id for fav in st.session_state.favorite_images)
            if st.button(
                "⭐" if is_fav else "☆", 
                key=f"fav_{image_id}", 
                use_container_width=True, 
                help="收藏/取消收藏"
            ):
                if is_fav: 
                    st.session_state.favorite_images = [
                        f for f in st.session_state.favorite_images 
                        if f['id'] != image_id
                    ]
                else: 
                    st.session_state.favorite_images.append({
                        "id": image_id, 
                        "image_b64": b64_json, 
                        "timestamp": datetime.datetime.now(), 
                        "history_item": history_item
                    })
                rerun_app()
                
        with col3:
            if st.button(
                "🎨 變體", 
                key=f"vary_{image_id}", 
                use_container_width=True, 
                help="使用此提示生成變體"
            ):
                st.session_state.update({
                    'vary_prompt': history_item['prompt'], 
                    'vary_negative_prompt': history_item.get('negative_prompt', ''), 
                    'vary_model': history_item['model']
                })
                rerun_app()
                
    except Exception as e: 
        st.error(f"圖像顯示錯誤: {e}")

def init_api_client():
    cfg = get_active_config()
    if cfg and cfg.get('api_key') and cfg.get('provider') not in ["Pollinations.ai", "Hugging Face"]:
        try: 
            return OpenAI(api_key=cfg['api_key'], base_url=cfg['base_url'])
        except Exception: 
            return None
    return None

def editor_provider_changed():
    provider = st.session_state.editor_provider_selectbox
    st.session_state.editor_base_url = API_PROVIDERS[provider]['base_url_default']
    st.session_state.editor_api_key = ""

def load_profile_to_editor_state(profile_name):
    config = st.session_state.api_profiles.get(profile_name, {})
    provider = config.get('provider', 'Pollinations.ai')
    
    st.session_state.editor_provider_selectbox = provider
    st.session_state.editor_base_url = config.get(
        'base_url', 
        API_PROVIDERS.get(provider, {}).get('base_url_default', '')
    )
    st.session_state.editor_api_key = config.get('api_key', '')
    st.session_state.editor_auth_mode = config.get('pollinations_auth_mode', '免費')
    st.session_state.editor_referrer = config.get('pollinations_referrer', '')
    st.session_state.editor_token = config.get('pollinations_token', '')
    st.session_state.profile_being_edited = profile_name

def show_api_settings():
    st.subheader("⚙️ API 存檔管理")
    
    profile_names = list(st.session_state.api_profiles.keys())
    if not profile_names: 
        st.warning("沒有可用的 API 存檔。請新增一個。")
        
    active_profile_name = st.selectbox(
        "活動存檔", 
        profile_names, 
        index=profile_names.index(st.session_state.get('active_profile_name')) 
        if st.session_state.get('active_profile_name') in profile_names else 0
    )
    
    if (st.session_state.get('active_profile_name') != active_profile_name or 
        'profile_being_edited' not in st.session_state or 
        st.session_state.profile_being_edited != active_profile_name):
        
        st.session_state.active_profile_name = active_profile_name
        load_profile_to_editor_state(active_profile_name)
        st.session_state.discovered_models = {}
        rerun_app()

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➕ 新增存檔", use_container_width=True):
            new_name = "新存檔"
            count = 1
            while new_name in st.session_state.api_profiles: 
                new_name = f"新存檔_{count}"
                count += 1
                
            st.session_state.api_profiles[new_name] = {
                'provider': 'Pollinations.ai', 
                'validated': False, 
                'base_url': API_PROVIDERS['Pollinations.ai']['base_url_default']
            }
            st.session_state.active_profile_name = new_name
            rerun_app()
            
    with col2:
        if st.button(
            "🗑️ 刪除當前存檔", 
            use_container_width=True, 
            disabled=len(profile_names) <= 1 or not active_profile_name
        ):
            if active_profile_name:
                del st.session_state.api_profiles[active_profile_name]
                st.session_state.active_profile_name = list(st.session_state.api_profiles.keys())[0]
                rerun_app()

    if active_profile_name:
        with st.expander("📝 編輯當前活動存檔", expanded=True):
            st.text_input("存檔名稱", value=active_profile_name, key="editor_profile_name")
            
            st.selectbox(
                "API 提供商", 
                list(API_PROVIDERS.keys()), 
                key='editor_provider_selectbox', 
                on_change=editor_provider_changed
            )
            
            st.text_input("API 端點 URL", key='editor_base_url')
            
            provider = st.session_state.editor_provider_selectbox
            
            if provider == "Pollinations.ai":
                st.radio(
                    "認證模式", 
                    ["免費", "域名", "令牌"], 
                    key='editor_auth_mode', 
                    horizontal=True
                )
                st.text_input(
                    "應用域名 (Referrer)", 
                    key='editor_referrer', 
                    disabled=(st.session_state.editor_auth_mode != '域名')
                )
                st.text_input(
                    "API 令牌 (Token)", 
                    key='editor_token', 
                    type="password", 
                    disabled=(st.session_state.editor_auth_mode != '令牌')
                )
            else: 
                st.text_input("API 密鑰", key='editor_api_key', type="password")

            if st.button("💾 保存/更新存檔", type="primary"):
                provider = st.session_state.editor_provider_selectbox
                new_config = {
                    'provider': provider, 
                    'base_url': st.session_state.editor_base_url
                }
                
                if provider == "Pollinations.ai":
                    new_config.update({
                        'api_key': '', 
                        'pollinations_auth_mode': st.session_state.editor_auth_mode, 
                        'pollinations_referrer': st.session_state.editor_referrer, 
                        'pollinations_token': st.session_state.editor_token
                    })
                else: 
                    new_config.update({
                        'api_key': st.session_state.editor_api_key, 
                        'pollinations_auth_mode': '免費', 
                        'pollinations_referrer': '', 
                        'pollinations_token': ''
                    })
                    
                is_valid, msg = validate_api_key(
                    new_config['api_key'], 
                    new_config['base_url'], 
                    new_config['provider']
                )
                new_config['validated'] = is_valid
                
                new_name = st.session_state.editor_profile_name
                if new_name != active_profile_name: 
                    del st.session_state.api_profiles[active_profile_name]
                    
                st.session_state.api_profiles[new_name] = new_config
                st.session_state.active_profile_name = new_name
                
                st.success(f"存檔 '{new_name}' 已保存。狀態: {msg}")
                time.sleep(1)
                rerun_app()

def show_model_selector(all_models):
    """顯示分類的模型選擇器"""
    categorized_models = get_models_by_category(all_models)
    
    # 獲取默認值
    prompt_default = st.session_state.pop('vary_prompt', '')
    neg_prompt_default = st.session_state.pop('vary_negative_prompt', '')
    model_default_key = st.session_state.pop('vary_model', list(all_models.keys())[0])
    
    st.subheader("🤖 模型選擇")
    
    # 顯示模型統計
    total_models = len(all_models)
    categories_count = len(categorized_models)
    st.caption(f"可用模型: {total_models} 個，分為 {categories_count} 個類別")
    
    selected_model = None
    
    # 按類別顯示模型
    for category, models in categorized_models.items():
        # 展開/收合狀態
        expanded_key = f"category_{category}_expanded"
        if expanded_key not in st.session_state:
            st.session_state[expanded_key] = category in ['FLUX', 'Stable Diffusion']
            
        with st.expander(f"📁 {category} ({len(models)} 個模型)", expanded=st.session_state[expanded_key]):
            # 創建網格布局
            cols = st.columns(3)
            for i, (model_id, model_info) in enumerate(models.items()):
                col = cols[i % 3]
                with col:
                    model_name = f"{model_info.get('icon', '🤖')} {model_info.get('name', model_id)}"
                    if st.button(
                        model_name, 
                        key=f"select_model_{model_id}",
                        use_container_width=True,
                        type="primary" if model_id == model_default_key else "secondary"
                    ):
                        selected_model = model_id
                        st.session_state.selected_model = model_id
                        rerun_app()
    
    # 返回選中的模型
    if selected_model:
        return selected_model
    elif 'selected_model' in st.session_state and st.session_state.selected_model in all_models:
        return st.session_state.selected_model
    else:
        return model_default_key if model_default_key in all_models else list(all_models.keys())[0]

# 初始化
init_session_state()
client = init_api_client()
cfg = get_active_config()
api_configured = cfg and cfg.get('validated', False)

# --- 側邊欄 ---
with st.sidebar:
    show_api_settings()
    st.markdown("---")
    
    if api_configured:
        st.success(f"🟢 活動存檔: '{st.session_state.active_profile_name}'")
        
        # 顯示當前API供應商信息
        provider_info = API_PROVIDERS.get(cfg['provider'], {})
        st.info(f"{provider_info.get('icon', '🤖')} {provider_info.get('name', cfg['provider'])}")
        
        can_discover = (client is not None) or (cfg.get('provider') in ["Pollinations.ai", "Hugging Face"])
        
        if st.button("🔍 發現模型", use_container_width=True, disabled=not can_discover):
            with st.spinner("🔍 正在發現模型..."):
                discovered = auto_discover_models(client, cfg['provider'], cfg['base_url'])
                st.session_state.discovered_models = discovered
                if discovered:
                    st.success(f"發現 {len(discovered)} 個模型！")
                else:
                    st.warning("未發現任何模型。")
                time.sleep(1)
                rerun_app()
                
    elif st.session_state.api_profiles: 
        st.error(f"🔴 '{st.session_state.active_profile_name}' 未驗證")
        
    st.markdown("---")
    st.info(f"⚡ **增強版優化**\n- 歷史: {MAX_HISTORY_ITEMS}\n- 收藏: {MAX_FAVORITE_ITEMS}\n- 批量: {MAX_BATCH_SIZE}")

# --- 主標題 ---
st.title("🎨 AI 圖像生成器 (多模型增強版)")
st.caption("支援 FLUX、Stable Diffusion、DALL-E 及更多模型")

# --- 主介面 ---
tab1, tab2, tab3 = st.tabs([
    "🚀 生成圖像", 
    f"📚 歷史 ({len(st.session_state.generation_history)})", 
    f"⭐ 收藏 ({len(st.session_state.favorite_images)})"
])

with tab1:
    if not api_configured: 
        st.warning("⚠️ 請在側邊欄選擇一個已驗證的存檔，或新增一個。")
    else:
        all_models = merge_models()
        if not all_models: 
            st.warning("⚠️ 未發現任何模型。請點擊側邊欄的「發現模型」。")
        else:
            # 模型選擇
            selected_model = show_model_selector(all_models)
            
            # 顯示當前選中的模型
            if selected_model:
                model_info = all_models[selected_model]
                st.success(f"已選擇模型: {model_info.get('icon', '🤖')} {model_info.get('name', selected_model)}")
            
            st.markdown("---")
            
            # 生成參數
            col1, col2 = st.columns([2, 1])
            
            with col1:
                prompt_default = st.session_state.get('vary_prompt', '')
                neg_prompt_default = st.session_state.get('vary_negative_prompt', '')
                
                selected_style = st.selectbox("🎨 風格預設:", list(STYLE_PRESETS.keys()))
                
                prompt_val = st.text_area(
                    "✍️ 提示詞:", 
                    value=prompt_default, 
                    height=100, 
                    placeholder="一隻貓在日落下飛翔，電影感，高品質"
                )
                
                negative_prompt_val = st.text_area(
                    "🚫 負向提示詞:", 
                    value=neg_prompt_default, 
                    height=50, 
                    placeholder="模糊, 糟糕的解剖結構, 文字, 水印"
                )
                
            with col2:
                n_images = st.slider("生成數量", 1, MAX_BATCH_SIZE, 1)
                
                size_preset = st.selectbox(
                    "圖像尺寸", 
                    options=list(IMAGE_SIZES.keys()), 
                    format_func=lambda x: IMAGE_SIZES[x]
                )
                
                final_size_str = size_preset
                if size_preset == "自定義...":
                    width = st.slider("寬度", 256, 2048, 1024, 64)
                    height = st.slider("高度", 256, 2048, 1024, 64)
                    final_size_str = f"{width}x{height}"
            
            # API 特定選項
            enhance, private, nologo, safe = False, False, False, False
            
            if cfg.get('provider') == "Pollinations.ai":
                with st.expander("🌸 Pollinations.ai 進階選項"):
                    col1, col2 = st.columns(2)
                    with col1:
                        enhance = st.checkbox("增強提示詞", True)
                        private = st.checkbox("私密模式", True)
                    with col2:
                        nologo = st.checkbox("移除標誌", True)
                        safe = st.checkbox("安全模式", False)
            
            elif cfg.get('provider') == "Hugging Face":
                with st.expander("🤗 Hugging Face 進階選項"):
                    col1, col2 = st.columns(2)
                    with col1:
                        inference_steps = st.slider("推理步驟", 10, 50, 20)
                        guidance_scale = st.slider("引導強度", 1.0, 20.0, 7.5, 0.5)
                    with col2:
                        scheduler = st.selectbox("調度器", ["DPMSolverMultistep", "EulerDiscrete", "DDIM"])
            
            # 生成按鈕
            if st.button(
                "🚀 生成圖像", 
                type="primary", 
                use_container_width=True, 
                disabled=not prompt_val.strip() or not selected_model
            ):
                final_prompt = (
                    f"{prompt_val}, {STYLE_PRESETS[selected_style]}" 
                    if selected_style != "無" and STYLE_PRESETS[selected_style] 
                    else prompt_val
                )
                
                with st.spinner(f"🎨 正在使用 {all_models[selected_model]['name']} 生成 {n_images} 張圖像..."):
                    params = {
                        "model": selected_model, 
                        "prompt": final_prompt, 
                        "negative_prompt": negative_prompt_val, 
                        "size": final_size_str, 
                        "n": n_images, 
                        "enhance": enhance, 
                        "private": private, 
                        "nologo": nologo, 
                        "safe": safe
                    }
                    
                    success, result = generate_images_with_retry(client, **params)
                    
                    if success and result.data:
                        img_b64s = [img.b64_json for img in result.data]
                        add_to_history(
                            prompt_val, 
                            negative_prompt_val, 
                            selected_model, 
                            img_b64s, 
                            {
                                "size": final_size_str, 
                                "provider": cfg['provider'], 
                                "style": selected_style, 
                                "n": n_images,
                                "model_name": all_models[selected_model]['name']
                            }
                        )
                        
                        st.success(f"✨ 成功生成 {len(img_b64s)} 張圖像！")
                        
                        # 顯示生成的圖像
                        cols = st.columns(min(len(img_b64s), 2))
                        for i, b64_json in enumerate(img_b64s):
                            with cols[i % 2]: 
                                display_image_with_actions(
                                    b64_json, 
                                    f"{st.session_state.generation_history[0]['id']}_{i}", 
                                    st.session_state.generation_history[0]
                                )
                        
                        gc.collect()
                    else: 
                        st.error(f"❌ 生成失敗: {result}")

with tab2:
    if not st.session_state.generation_history: 
        st.info("📭 尚無生成歷史。")
    else:
        for item in st.session_state.generation_history:
            timestamp_str = item['timestamp'].strftime('%m-%d %H:%M')
            model_name = all_models.get(item['model'], {}).get('name', item['model']) if 'all_models' in locals() else item['model']
            
            with st.expander(f"🎨 {item['prompt'][:50]}... | {model_name} | {timestamp_str}"):
                st.markdown(f"**提示詞**: {item['prompt']}")
                st.markdown(f"**模型**: {model_name}")
                
                if item.get('negative_prompt'): 
                    st.markdown(f"**負向提示詞**: {item['negative_prompt']}")
                    
                if item.get('metadata', {}).get('style'):
                    st.markdown(f"**風格**: {item['metadata']['style']}")
                    
                cols = st.columns(min(len(item['images']), 2))
                for i, b64_json in enumerate(item['images']):
                    with cols[i % 2]: 
                        display_image_with_actions(b64_json, f"hist_{item['id']}_{i}", item)

with tab3:
    if not st.session_state.favorite_images: 
        st.info("⭐ 尚無收藏的圖像。")
    else:
        # 收藏管理
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"⭐ 我的收藏 ({len(st.session_state.favorite_images)} 張)")
        with col2:
            if st.button("🗑️ 清空收藏", use_container_width=True):
                st.session_state.favorite_images = []
                rerun_app()
        
        # 顯示收藏的圖像
        cols = st.columns(3)
        for i, fav in enumerate(sorted(st.session_state.favorite_images, key=lambda x: x['timestamp'], reverse=True)):
            with cols[i % 3]: 
                display_image_with_actions(
                    fav['image_b64'], 
                    fav['id'], 
                    fav.get('history_item', {})
                )

# --- 頁腳 ---
st.markdown("---")
st.markdown(
    """<div style="text-align: center; color: #888; margin-top: 2rem;">
    <small>🎨 多模型增強版 | 支援 FLUX、Stable Diffusion、DALL-E 等 | 部署在雲端平台 🎨</small>
    </div>""", 
    unsafe_allow_html=True
)
