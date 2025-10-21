import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import datetime
import base64
from typing import Dict, List, Tuple, Optional
import time
import random
import uuid
import os
from urllib.parse import urlencode, quote
import gc
from streamlit.errors import StreamlitAPIException

# 應用配置
APP_TITLE = "🎨 AI 圖像生成器 (多模型版)"
VERSION = "v1.5.0"

# 限制配置
MAX_HISTORY_ITEMS = 20
MAX_FAVORITE_ITEMS = 40
MAX_BATCH_SIZE = 4
REQUEST_TIMEOUT = 120

# 圖像尺寸預設
IMAGE_SIZES = {
    "自定義...": "Custom",
    "512x512": "SD 標準 (1:1)", 
    "768x768": "SD XL 標準 (1:1)",
    "1024x1024": "正方形 (1:1)", 
    "1080x1080": "IG 貼文 (1:1)",
    "512x768": "SD 縱向 (2:3)",
    "768x1024": "SDXL 縱向 (3:4)",
    "1080x1350": "IG 縱向 (4:5)", 
    "1080x1920": "IG Story (9:16)",
    "768x512": "SD 橫向 (3:2)",
    "1024x768": "SDXL 橫向 (4:3)",
    "1200x630": "FB 橫向 (1.91:1)",
    "1536x640": "超寬橫幅 (2.4:1)",
}

# 風格預設
STYLE_PRESETS = {
    "無": "",
    "電影感": "cinematic, dramatic lighting, high detail, sharp focus, epic scene",
    "動漫風": "anime, manga style, vibrant colors, clean line art, studio ghibli", 
    "賽博龐克": "cyberpunk, neon lights, futuristic city, high-tech, Blade Runner",
    "人像攝影": "portrait photography, professional headshot, studio lighting, bokeh",
    "街頭攝影": "street photography, candid moment, urban setting, natural lighting",
    "風景攝影": "landscape photography, golden hour lighting, wide angle view, HDR",
    "印象派": "impressionism, soft brushstrokes, natural light, Monet style",
    "超現實主義": "surrealism, dreamlike imagery, Salvador Dali style",
    "普普藝術": "pop art, bold colors, comic book style, Andy Warhol",
    "水墨畫": "traditional Chinese ink painting, minimalist zen aesthetic",
    "水彩畫": "watercolor painting, soft transparent washes, delicate colors",
    "3D 渲染": "3D render, octane rendering, photorealistic, volumetric lighting",
    "像素藝術": "pixel art, 8-bit style, retro gaming aesthetic",
    "蒸汽龐克": "steampunk aesthetic, Victorian era meets technology",
    "奇幻藝術": "fantasy art, magical creatures, epic landscapes",
    "科幻藝術": "science fiction art, futuristic technology, space scenes",
    "美式漫畫": "American comic book style, bold outlines, dynamic poses",
    "日式漫畫": "manga style, detailed line art, expressive characters",
    "黑白攝影": "black and white photography, high contrast, dramatic shadows",
    "矢量圖": "vector illustration, clean geometric lines, flat design",
    "油畫": "oil painting, thick impasto, rich textures, renaissance style",
    "素描": "pencil sketch, graphite drawing, crosshatching, detailed line work",
    "包豪斯": "Bauhaus design, geometric minimalism, functional aesthetics",
    "裝飾藝術": "art deco style, geometric patterns, luxury aesthetics",
}

# 負向提示詞預設
NEGATIVE_PROMPTS = {
    "基本": "blurry, low quality, distorted, deformed, ugly, bad anatomy",
    "攝影": "blurry, low resolution, overexposed, underexposed, noise",
    "人像": "bad anatomy, deformed face, extra limbs, missing fingers",
    "動漫": "realistic, photographic, 3d render, western cartoon",
    "藝術": "photographic, realistic, low quality, commercial",
}

def rerun_app():
    try:
        if hasattr(st, 'rerun'):
            st.rerun()
        elif hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            st.stop()
    except Exception:
        st.stop()

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API供應商配置
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
            "dalle-3": {"name": "DALL-E 3", "icon": "🤖", "category": "Professional"},
            "playground-v2.5": {"name": "Playground v2.5", "icon": "🎪", "category": "Professional"},
            # 社區模型
            "dreamshaper": {"name": "DreamShaper", "icon": "💫", "category": "Community"},
            "realistic-vision": {"name": "Realistic Vision", "icon": "👁️", "category": "Community"},
            "deliberate": {"name": "Deliberate", "icon": "🎨", "category": "Community"},
            "anything-v5": {"name": "Anything v5", "icon": "🌟", "category": "Anime"},
            "waifu-diffusion": {"name": "Waifu Diffusion", "icon": "👩‍🎨", "category": "Anime"},
            "openjourney": {"name": "OpenJourney", "icon": "🗺️", "category": "Community"},
            # 風格模型
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
    "Hugging Face": {
        "name": "Hugging Face Inference",
        "base_url_default": "https://api-inference.huggingface.co",
        "icon": "🤗",
        "hardcoded_models": {
            "stable-diffusion-v1-5": {"name": "SD 1.5 (HF)", "icon": "🔰", "category": "Stable Diffusion"},
            "stable-diffusion-xl-base-1.0": {"name": "SDXL Base (HF)", "icon": "💎", "category": "Stable Diffusion"},
            "flux-1-dev": {"name": "Flux.1 Dev (HF)", "icon": "🛠️", "category": "FLUX"},
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
    }
}

def init_session_state():
    if 'api_profiles' not in st.session_state:
        try:
            base_profiles = st.secrets.get("api_profiles", {})
        except:
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
    
    if ('active_profile_name' not in st.session_state or 
        st.session_state.active_profile_name not in st.session_state.api_profiles):
        st.session_state.active_profile_name = (
            list(st.session_state.api_profiles.keys())[0] 
            if st.session_state.api_profiles else ""
        )
    
    defaults = {
        'generation_history': [],
        'favorite_images': [],
        'discovered_models': {},
        'selected_model': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_active_config():
    return st.session_state.api_profiles.get(st.session_state.active_profile_name, {})

def validate_api_key(api_key: str, base_url: str, provider: str) -> Tuple[bool, str]:
    try:
        if provider == "Pollinations.ai":
            return True, "Pollinations.ai 無需驗證"
        
        elif provider == "Hugging Face":
            if not api_key:
                return False, "Hugging Face 需要 API Token"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
            if response.status_code == 200:
                return True, "Hugging Face API Token 驗證成功"
            else:
                return False, f"Hugging Face API 驗證失敗: {response.status_code}"
        
        else:
            client = OpenAI(api_key=api_key, base_url=base_url)
            client.models.list()
            return True, "API 密鑰驗證成功"
            
    except Exception as e:
        return False, f"API 驗證失敗: {str(e)[:100]}"

def get_models_by_category(models: Dict[str, Dict]) -> Dict[str, Dict[str, Dict]]:
    categorized = {}
    for model_id, model_info in models.items():
        category = model_info.get('category', 'Other')
        if category not in categorized:
            categorized[category] = {}
        categorized[category][model_id] = model_info
    
    priority_order = ["FLUX", "Stable Diffusion", "Professional", "Anime", "Style", "Community", "Other"]
    sorted_categorized = {}
    
    for category in priority_order:
        if category in categorized:
            sorted_categorized[category] = categorized[category]
    
    for category, models in categorized.items():
        if category not in sorted_categorized:
            sorted_categorized[category] = models
    
    return sorted_categorized

def merge_models() -> Dict[str, Dict]:
    provider = get_active_config().get('provider')
    discovered = st.session_state.get('discovered_models', {})
    
    if provider in API_PROVIDERS:
        hardcoded = API_PROVIDERS[provider].get('hardcoded_models', {})
        merged = {**hardcoded, **discovered}
    else:
        merged = discovered
    
    return merged

def generate_images_with_retry(client, **params) -> Tuple[bool, any]:
    provider = get_active_config().get('provider')
    n_images = params.get("n", 1)
    
    if provider == "Pollinations.ai":
        return generate_pollinations_images(params, n_images)
    elif provider == "Hugging Face":
        return generate_huggingface_images(params, n_images)
    else:
        return generate_openai_compatible_images(client, params, n_images)

def generate_pollinations_images(params: Dict, n_images: int) -> Tuple[bool, any]:
    generated_images = []
    cfg = get_active_config()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n_images):
        try:
            status_text.text(f"正在生成第 {i+1}/{n_images} 張圖片...")
            progress_bar.progress(i / n_images)
            
            current_params = params.copy()
            current_params["seed"] = random.randint(0, 2**32 - 1)
            
            prompt = current_params.get("prompt", "")
            if neg_prompt := current_params.get("negative_prompt"):
                prompt += f" --no {neg_prompt}"
            
            width, height = str(current_params.get("size", "1024x1024")).split('x')
            
            api_params = {}
            for key, value in {
                "model": current_params.get("model"),
                "width": width,
                "height": height,
                "seed": current_params.get("seed"),
                "nologo": current_params.get("nologo"),
                "private": current_params.get("private"),
                "enhance": current_params.get("enhance"),
                "safe": current_params.get("safe")
            }.items():
                if value is not None:
                    api_params[key] = value
            
            headers = {}
            auth_mode = cfg.get('pollinations_auth_mode', '免費')
            
            if auth_mode == '令牌' and cfg.get('pollinations_token'):
                headers['Authorization'] = f"Bearer {cfg['pollinations_token']}"
            elif auth_mode == '域名' and cfg.get('pollinations_referrer'):
                headers['Referer'] = cfg['pollinations_referrer']
            
            url = f"{cfg['base_url']}/prompt/{quote(prompt)}?{urlencode(api_params)}"
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            
            if response.ok:
                b64_json = base64.b64encode(response.content).decode()
                image_obj = type('Image', (object,), {'b64_json': b64_json})
                generated_images.append(image_obj)
            else:
                st.warning(f"第 {i+1} 張圖片生成失敗: HTTP {response.status_code}")
                
        except Exception as e:
            st.warning(f"第 {i+1} 張圖片生成錯誤: {str(e)[:100]}")
            continue
    
    progress_bar.progress(1.0)
    status_text.text(f"完成生成 {len(generated_images)}/{n_images} 張圖片")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    if generated_images:
        response_obj = type('Response', (object,), {'data': generated_images})
        return True, response_obj
    else:
        return False, "所有圖片生成均失敗"

def generate_huggingface_images(params: Dict, n_images: int) -> Tuple[bool, any]:
    generated_images = []
    cfg = get_active_config()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n_images):
        try:
            status_text.text(f"正在通過HF生成第 {i+1}/{n_images} 張圖片...")
            progress_bar.progress(i / n_images)
            
            headers = {"Authorization": f"Bearer {cfg['api_key']}"}
            model = params.get("model")
            prompt = params.get("prompt", "")
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": params.get("negative_prompt", ""),
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                }
            }
            
            url = f"{cfg['base_url']}/models/{model}"
            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            
            if response.ok:
                b64_json = base64.b64encode(response.content).decode()
                image_obj = type('Image', (object,), {'b64_json': b64_json})
                generated_images.append(image_obj)
            else:
                st.warning(f"第 {i+1} 張圖片生成失敗: HTTP {response.status_code}")
                
        except Exception as e:
            st.warning(f"第 {i+1} 張圖片生成錯誤: {str(e)[:100]}")
            continue
    
    progress_bar.progress(1.0)
    status_text.text(f"完成生成 {len(generated_images)}/{n_images} 張圖片")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    if generated_images:
        response_obj = type('Response', (object,), {'data': generated_images})
        return True, response_obj
    else:
        return False, "所有圖片生成均失敗"

def generate_openai_compatible_images(client, params: Dict, n_images: int) -> Tuple[bool, any]:
    try:
        sdk_params = {
            "model": params.get("model"),
            "prompt": params.get("prompt"),
            "size": str(params.get("size")),
            "n": n_images,
            "response_format": "b64_json"
        }
        
        if params.get("negative_prompt"):
            sdk_params["negative_prompt"] = params.get("negative_prompt")
        
        sdk_params = {k: v for k, v in sdk_params.items() if v is not None and v != ""}
        return True, client.images.generate(**sdk_params)
    except Exception as e:
        return False, str(e)[:200]

def add_to_history(prompt: str, negative_prompt: str, model: str, images: List[str], metadata: Dict):
    history = st.session_state.generation_history
    new_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now(),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model": model,
        "images": images,
        "metadata": metadata
    }
    history.insert(0, new_entry)
    st.session_state.generation_history = history[:MAX_HISTORY_ITEMS]

def display_image_with_actions(b64_json: str, image_id: str, history_item: Dict):
    try:
        img_data = base64.b64decode(b64_json)
        img = Image.open(BytesIO(img_data))
        st.image(img, use_container_width=True)
        
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
                    if len(st.session_state.favorite_images) < MAX_FAVORITE_ITEMS:
                        st.session_state.favorite_images.append({
                            "id": image_id,
                            "image_b64": b64_json,
                            "timestamp": datetime.datetime.now(),
                            "history_item": history_item
                        })
                    else:
                        st.warning(f"收藏已達上限 ({MAX_FAVORITE_ITEMS})")
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
        st.error(f"圖像顯示錯誤: {str(e)[:100]}")

def init_api_client():
    cfg = get_active_config()
    if (cfg and cfg.get('api_key') and cfg.get('provider') not in ["Pollinations.ai", "Hugging Face"]):
        try:
            return OpenAI(api_key=cfg['api_key'], base_url=cfg['base_url'])
        except Exception:
            return None
    return None

def show_model_selector(all_models: Dict[str, Dict]) -> Optional[str]:
    if not all_models:
        st.warning("⚠️ 沒有可用的模型。請在側邊欄配置API。")
        return None
    
    categorized_models = get_models_by_category(all_models)
    current_selection = st.session_state.get('selected_model')
    
    if current_selection not in all_models:
        current_selection = list(all_models.keys())[0]
        st.session_state.selected_model = current_selection
    
    if current_selection:
        model_info = all_models[current_selection]
        st.success(
            f"🎯 當前模型: {model_info.get('icon', '🤖')} "
            f"{model_info.get('name', current_selection)}"
        )
    
    st.markdown("---")
    total_models = len(all_models)
    categories_count = len(categorized_models)
    st.caption(f"📊 可用模型: **{total_models}** 個，分為 **{categories_count}** 個類別")
    
    for category, models in categorized_models.items():
        with st.expander(f"📁 {category} ({len(models)} 個模型)", expanded=True):
            cols = st.columns(3)
            
            for i, (model_id, model_info) in enumerate(models.items()):
                col = cols[i % 3]
                
                with col:
                    button_text = f"{model_info.get('icon', '🤖')} {model_info.get('name', model_id)}"
                    is_selected = st.session_state.get('selected_model') == model_id
                    
                    if st.button(
                        button_text,
                        key=f"select_model_{model_id}_{category}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        st.session_state.selected_model = model_id
                        rerun_app()
    
    return current_selection

def main():
    init_session_state()
    client = init_api_client()
    cfg = get_active_config()
    api_configured = cfg and cfg.get('validated', False)
    
    # 側邊欄
    with st.sidebar:
        st.subheader("⚙️ API 設置")
        
        if api_configured:
            provider_info = API_PROVIDERS.get(cfg['provider'], {})
            st.success(f"🟢 已連接: {st.session_state.active_profile_name}")
            st.info(f"{provider_info.get('icon', '🤖')} {provider_info.get('name', cfg['provider'])}")
        else:
            st.warning("⚠️ 請配置API供應商")
        
        st.markdown("---")
        st.info(f"""
        **📊 統計信息**
        - 歷史: {len(st.session_state.generation_history)}/{MAX_HISTORY_ITEMS}
        - 收藏: {len(st.session_state.favorite_images)}/{MAX_FAVORITE_ITEMS}
        - 批次上限: {MAX_BATCH_SIZE}
        """)
    
    # 主標題
    st.title(APP_TITLE)
    st.caption(f"支援 FLUX、Stable Diffusion、DALL-E 及更多模型 | {VERSION}")
    
    # 主界面
    tab1, tab2, tab3 = st.tabs([
        "🚀 生成圖像",
        f"📚 歷史 ({len(st.session_state.generation_history)})",
        f"⭐ 收藏 ({len(st.session_state.favorite_images)})"
    ])
    
    with tab1:
        if not api_configured:
            st.warning("⚠️ 請在側邊欄配置並驗證API供應商")
        else:
            all_models = merge_models()
            selected_model = show_model_selector(all_models)
            
            if selected_model:
                st.markdown("---")
                
                # 生成參數
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    selected_style = st.selectbox(
                        "🎨 風格預設",
                        list(STYLE_PRESETS.keys())
                    )
                    
                    prompt_val = st.text_area(
                        "✍️ 提示詞",
                        value=st.session_state.pop('vary_prompt', ''),
                        height=120,
                        placeholder="描述您想要生成的圖像..."
                    )
                    
                    neg_preset = st.selectbox(
                        "🚫 負向提示詞預設",
                        list(NEGATIVE_PROMPTS.keys())
                    )
                    
                    negative_prompt_val = st.text_area(
                        "🚫 負向提示詞",
                        value=st.session_state.pop('vary_negative_prompt', '') or NEGATIVE_PROMPTS.get(neg_preset, ""),
                        height=80,
                        placeholder="不想要的內容..."
                    )
                
                with col2:
                    n_images = st.slider(
                        "🖼️ 生成數量",
                        1, MAX_BATCH_SIZE, 1
                    )
                    
                    size_preset = st.selectbox(
                        "📐 圖像尺寸",
                        options=list(IMAGE_SIZES.keys()),
                        format_func=lambda x: IMAGE_SIZES[x]
                    )
                    
                    if size_preset == "自定義...":
                        col_w, col_h = st.columns(2)
                        with col_w:
                            width = st.slider("寬度", 256, 2048, 1024, 64)
                        with col_h:
                            height = st.slider("高度", 256, 2048, 1024, 64)
                        final_size_str = f"{width}x{height}"
                    else:
                        final_size_str = size_preset
                
                # 高級選項
                advanced_options = {}
                if cfg.get('provider') == "Pollinations.ai":
                    with st.expander("🌸 Pollinations.ai 進階選項"):
                        col1, col2 = st.columns(2)
                        with col1:
                            advanced_options['enhance'] = st.checkbox("✨ 增強提示詞", True)
                            advanced_options['private'] = st.checkbox("🔒 私密模式", True)
                        with col2:
                            advanced_options['nologo'] = st.checkbox("🚫 移除標誌", True)
                            advanced_options['safe'] = st.checkbox("🛡️ 安全模式", False)
                
                # 生成按鈕
                if st.button(
                    "🚀 生成圖像",
                    type="primary",
                    use_container_width=True,
                    disabled=not prompt_val.strip()
                ):
                    final_prompt = prompt_val
                    if selected_style != "無" and STYLE_PRESETS[selected_style]:
                        final_prompt = f"{final_prompt}, {STYLE_PRESETS[selected_style]}"
                    
                    params = {
                        "model": selected_model,
                        "prompt": final_prompt,
                        "negative_prompt": negative_prompt_val,
                        "size": final_size_str,
                        "n": n_images,
                        **advanced_options
                    }
                    
                    model_name = all_models[selected_model]['name']
                    with st.spinner(f"🎨 正在使用 {model_name} 生成 {n_images} 張圖像..."):
                        success, result = generate_images_with_retry(client, **params)
                    
                    if success and hasattr(result, 'data') and result.data:
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
                                "model_name": model_name
                            }
                        )
                        
                        st.success(f"✨ 成功生成 {len(img_b64s)} 張圖像！")
                        
                        if len(img_b64s) == 1:
                            display_image_with_actions(
                                img_b64s[0],
                                f"{st.session_state.generation_history[0]['id']}_0",
                                st.session_state.generation_history[0]
                            )
                        else:
                            cols = st.columns(2)
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
            st.info("📭 還沒有生成歷史。")
        else:
            for item in st.session_state.generation_history:
                timestamp_str = item['timestamp'].strftime('%m-%d %H:%M')
                all_models = merge_models()
                model_info = all_models.get(item['model'], {})
                model_name = model_info.get('name', item['model'])
                
                with st.expander(f"🎨 {item['prompt'][:60]}... | {model_name} | {timestamp_str}"):
                    st.markdown(f"**✍️ 提示詞:** {item['prompt']}")
                    if item.get('negative_prompt'):
                        st.markdown(f"**🚫 負向提示詞:** {item['negative_prompt']}")
                    
                    if len(item['images']) == 1:
                        display_image_with_actions(
                            item['images'][0],
                            f"hist_{item['id']}_0",
                            item
                        )
                    else:
                        cols = st.columns(2)
                        for i, b64_json in enumerate(item['images']):
                            with cols[i % 2]:
                                display_image_with_actions(
                                    b64_json,
                                    f"hist_{item['id']}_{i}",
                                    item
                                )
    
    with tab3:
        if not st.session_state.favorite_images:
            st.info("⭐ 還沒有收藏的圖像。")
        else:
            sorted_favorites = sorted(
                st.session_state.favorite_images,
                key=lambda x: x['timestamp'],
                reverse=True
            )
            
            cols = st.columns(3)
            for i, fav in enumerate(sorted_favorites):
                with cols[i % 3]:
                    display_image_with_actions(
                        fav['image_b64'],
                        fav['id'],
                        fav.get('history_item', {})
                    )
                    
                    fav_time = fav['timestamp'].strftime('%m-%d %H:%M')
                    st.caption(f"⭐ 收藏於: {fav_time}")
    
    # 頁腳
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #888; margin-top: 2rem;">
        <small>
            🎨 <strong>AI 圖像生成器 {VERSION}</strong> | 
            支持多模型和多供應商 | 
            讓創意無限延伸 🎨
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
