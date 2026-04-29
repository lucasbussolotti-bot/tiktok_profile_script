import os
import re
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ==============================
# CONFIG
# ==============================
SOCIAVAULT_API_KEY = os.environ.get("SOCIAVAULT_API_KEY", "")
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
GDRIVE_CREDENTIALS = os.environ.get("GDRIVE_CREDENTIALS", "")

SHEET_TIKTOK_PROFILE_ID  = "1947Wx86ZtNWQSaqcYVSXv_3WLvIA0p6u_Ol1DZ8GmX8"
SHEET_TT_DATA_PROFILE_ID = "1roDSHeO9-O_DKfTwUKAQv3euCUyfipKq_KxkpKpf3r4"
SHEET_TT_DATA_POST_ID    = "1o96u5EXkqhtxGdEqaUGYX4Us2HGnHfkVBLJIobqcma8"
SHEET_TT_DATA_COMMENTS_ID = "1shH8-PpUBTEuS7Izy4uTgmEcOHF-tdk_DbJR1ifXqJA"

TAB_TIKTOK_PROFILE  = "tiktok_profile"
TAB_TT_DATA_PROFILE = "tt_data_profile"
TAB_TT_DATA_POST    = "tt_data_post"
TAB_TT_DATA_COMMENTS = "tt_data_comments"

API_BASE         = "https://api.sociavault.com/v1/scrape/tiktok"
MAX_POSTS        = 12
POST_MAX_DAYS    = 14
GEMINI_BATCH     = 20
GEMINI_MAX_RETRY = 2

# ==============================
# GOOGLE SHEETS HELPERS
# ==============================

def get_google_service():
    creds_json = json.loads(GDRIVE_CREDENTIALS)
    creds = service_account.Credentials.from_service_account_info(
        creds_json,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=creds)


def read_sheet(service, spreadsheet_id, tab):
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"{tab}!A1:ZZ"
    ).execute()
    values = result.get("values", [])
    if not values:
        return pd.DataFrame()
    headers = values[0]
    rows = values[1:]
    # Pad rows shorter than headers
    rows = [r + [""] * (len(headers) - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=headers)


def append_to_sheet(service, spreadsheet_id, tab, df):
    if df.empty:
        return
    values = df.values.tolist()
    service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=f"{tab}!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": values}
    ).execute()


def ensure_header(service, spreadsheet_id, tab, columns):
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"{tab}!A1:1"
    ).execute()
    existing = result.get("values", [])
    if not existing:
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"{tab}!A1",
            valueInputOption="RAW",
            body={"values": [columns]}
        ).execute()

# ==============================
# SOCIAVAULT HELPERS
# ==============================

def sv_get(endpoint, params, timeout=60):
    headers = {"Authorization": f"Bearer {SOCIAVAULT_API_KEY}"}
    resp = requests.get(
        f"{API_BASE}/{endpoint}",
        headers=headers,
        params=params,
        timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()

# ==============================
# GEMINI HELPERS
# ==============================

def extrair_retry_seconds(error_str):
    match = re.search(r"retry in ([0-9.]+)s", error_str)
    if match:
        return float(match.group(1)) + 2
    return 60.0


def classify_comments_batch(client, comments_text):
    prompt = (
        "Você é um analista de redes sociais. Classifique cada comentário abaixo como "
        "'promotor' (positivo, elogio, apoio) ou 'detrator' (negativo, crítica, reclamação).\n"
        "Para cada comentário, retorne um JSON com os campos 'classification' e 'classification_reason'.\n"
        "Retorne APENAS uma lista JSON, sem markdown, sem texto extra.\n\n"
        "Comentários:\n"
    )
    for i, text in enumerate(comments_text):
        prompt += f"{i+1}. {text}\n"

    for attempt in range(1, GEMINI_MAX_RETRY + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            raw = response.text.strip()
            raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(raw)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = extrair_retry_seconds(err_str)
                if attempt < GEMINI_MAX_RETRY:
                    print(f"    Rate limit atingido. Aguardando {wait:.0f}s antes de tentar novamente (tentativa {attempt}/{GEMINI_MAX_RETRY})...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"    Rate limit após {GEMINI_MAX_RETRY} tentativas. Marcando lote como FALHA_API.", flush=True)
                    return [{"classification": "FALHA_API", "classification_reason": "rate limit"} for _ in comments_text]
            else:
                print(f"    Erro no Gemini: {e}", flush=True)
                return [{"classification": "ERRO", "classification_reason": str(e)} for _ in comments_text]

# ==============================
# ETAPA 1 — LER PERFIS
# ==============================

def ler_perfis(service):
    print("[ETAPA 1] Lendo perfis do tiktok_profile...", flush=True)
    df = read_sheet(service, SHEET_TIKTOK_PROFILE_ID, TAB_TIKTOK_PROFILE)
    if df.empty or "profile" not in df.columns:
        print("  Nenhum perfil encontrado.", flush=True)
        return []
    perfis = df[["profile", "date_added"]].dropna(subset=["profile"]).to_dict("records")
    print(f"  {len(perfis)} perfil(is) encontrado(s).", flush=True)
    return perfis

# ==============================
# ETAPA 2.0 — DADOS DO PERFIL
# ==============================

PROFILE_COLS = [
    "user_id", "username", "nickname", "verified",
    "followers", "following", "likes", "videos",
    "bio", "language", "is_organization", "run_datetime"
]

def processar_perfil(service, username):
    print(f"  [2.0] Buscando dados do perfil: {username}", flush=True)
    try:
        data = sv_get("profile", {"username": username})
    except Exception as e:
        print(f"    Erro ao buscar perfil {username}: {e}", flush=True)
        return None

    # Garante cabeçalho
    ensure_header(service, SHEET_TT_DATA_PROFILE_ID, TAB_TT_DATA_PROFILE, PROFILE_COLS)

    # Sempre insere nova linha para manter histórico de evolução
    row = {
        "user_id": str(data.get("user_id", data.get("id", ""))),
        "username": data.get("username", username),
        "nickname": data.get("nickname", ""),
        "verified": data.get("verified", ""),
        "followers": data.get("followers", data.get("follower_count", "")),
        "following": data.get("following", data.get("following_count", "")),
        "likes": data.get("likes", data.get("heart_count", "")),
        "videos": data.get("videos", data.get("video_count", "")),
        "bio": data.get("bio", data.get("signature", "")),
        "language": data.get("language", ""),
        "is_organization": data.get("is_organization", ""),
        "run_datetime": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    }
    df_row = pd.DataFrame([row])[PROFILE_COLS]
    append_to_sheet(service, SHEET_TT_DATA_PROFILE_ID, TAB_TT_DATA_PROFILE, df_row)
    print(f"    Perfil {username} salvo no tt_data_profile.", flush=True)
    return data

# ==============================
# ETAPA 2.1 — VÍDEOS / POSTS
# ==============================

POST_COLS = [
    "video_id", "description", "create_time", "author",
    "username", "followers", "likes", "comments",
    "views", "shares", "first_extracted_at", "video_url"
]

def processar_videos(service, username):
    print(f"  [2.1] Buscando vídeos de: {username}", flush=True)
    try:
        data = sv_get("videos", {"username": username, "limit": MAX_POSTS})
    except Exception as e:
        print(f"    Erro ao buscar vídeos de {username}: {e}", flush=True)
        return []

    videos = data if isinstance(data, list) else data.get("videos", data.get("items", []))
    videos = videos[:MAX_POSTS]

    if not videos:
        print(f"    Nenhum vídeo encontrado para {username}.", flush=True)
        return []

    # Garante cabeçalho
    ensure_header(service, SHEET_TT_DATA_POST_ID, TAB_TT_DATA_POST, POST_COLS)

    # Lê existentes para deduplicar por video_id
    existing_df = read_sheet(service, SHEET_TT_DATA_POST_ID, TAB_TT_DATA_POST)
    existing_ids = set(existing_df["video_id"].astype(str).tolist()) if not existing_df.empty and "video_id" in existing_df.columns else set()

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    novos = []

    for v in videos:
        video_id = str(v.get("video_id", v.get("id", "")))
        if video_id in existing_ids:
            continue

        video_url = v.get("video_url", v.get("url", f"https://www.tiktok.com/@{username}/video/{video_id}"))

        row = {
            "video_id": video_id,
            "description": v.get("description", v.get("desc", "")),
            "create_time": v.get("create_time", v.get("createTime", "")),
            "author": v.get("author", v.get("nickname", "")),
            "username": username,
            "followers": v.get("followers", v.get("follower_count", "")),
            "likes": v.get("likes", v.get("digg_count", "")),
            "comments": v.get("comments", v.get("comment_count", "")),
            "views": v.get("views", v.get("play_count", "")),
            "shares": v.get("shares", v.get("share_count", "")),
            "first_extracted_at": now_str,
            "video_url": video_url
        }
        novos.append(row)

    if novos:
        df_new = pd.DataFrame(novos)[POST_COLS]
        append_to_sheet(service, SHEET_TT_DATA_POST_ID, TAB_TT_DATA_POST, df_new)
        print(f"    {len(novos)} vídeo(s) novo(s) salvos para {username}.", flush=True)
    else:
        print(f"    Nenhum vídeo novo para {username}.", flush=True)

    # Retorna todos os 12 (inclusive já existentes) com video_url e first_extracted_at
    all_df = read_sheet(service, SHEET_TT_DATA_POST_ID, TAB_TT_DATA_POST)
    ids_perfil = [str(v.get("video_id", v.get("id", ""))) for v in videos]
    if not all_df.empty and "video_id" in all_df.columns:
        return all_df[all_df["video_id"].isin(ids_perfil)].to_dict("records")
    return []

# ==============================
# ETAPA 2.2 — COMENTÁRIOS
# ==============================

COMMENT_COLS = [
    "comment_id", "video_id", "text", "create_time",
    "likes", "replies_count", "purchase_intent",
    "user_name", "username", "language",
    "classification", "classification_reason"
]

def processar_comentarios(service, client, post):
    video_id  = str(post.get("video_id", ""))
    video_url = post.get("video_url", "")
    first_extracted = post.get("first_extracted_at", "")

    # Checar 14 dias
    try:
        extracted_dt = datetime.strptime(first_extracted, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        dias = (datetime.now(timezone.utc) - extracted_dt).days
        if dias > POST_MAX_DAYS:
            print(f"    Post {video_id} tem {dias} dias. Pulando comentários.", flush=True)
            return
    except Exception:
        pass

    print(f"    [2.2] Buscando comentários do vídeo: {video_url}", flush=True)

    # Lê IDs já salvos
    existing_df = read_sheet(service, SHEET_TT_DATA_COMMENTS_ID, TAB_TT_DATA_COMMENTS)
    existing_ids = set(existing_df["comment_id"].astype(str).tolist()) if not existing_df.empty and "comment_id" in existing_df.columns else set()

    # Garante cabeçalho
    ensure_header(service, SHEET_TT_DATA_COMMENTS_ID, TAB_TT_DATA_COMMENTS, COMMENT_COLS)

    # Buscar comentários
    try:
        data = sv_get("comments", {"url": video_url})
    except Exception as e:
        print(f"      Erro ao buscar comentários do vídeo {video_id}: {e}", flush=True)
        return

    comments = data if isinstance(data, list) else data.get("comments", data.get("items", []))

    novos = [c for c in comments if str(c.get("comment_id", c.get("id", ""))) not in existing_ids]
    if not novos:
        print(f"      Sem comentários novos para vídeo {video_id}.", flush=True)
        return

    print(f"      {len(novos)} comentário(s) novo(s) para classificar.", flush=True)

    # Classificar em lotes
    all_rows = []
    for i in range(0, len(novos), GEMINI_BATCH):
        lote = novos[i:i + GEMINI_BATCH]
        textos = [c.get("text", c.get("comment", "")) for c in lote]
        print(f"      Classificando lote {i // GEMINI_BATCH + 1}...", flush=True)
        classificacoes = classify_comments_batch(client, textos)

        for j, c in enumerate(lote):
            clf = classificacoes[j] if j < len(classificacoes) else {"classification": "ERRO", "classification_reason": "sem resposta"}
            row = {
                "comment_id": str(c.get("comment_id", c.get("id", ""))),
                "video_id": video_id,
                "text": c.get("text", c.get("comment", "")),
                "create_time": c.get("create_time", c.get("createTime", "")),
                "likes": c.get("likes", c.get("digg_count", "")),
                "replies_count": c.get("replies_count", c.get("reply_count", "")),
                "purchase_intent": c.get("purchase_intent", ""),
                "user_name": c.get("user_name", c.get("nickname", "")),
                "username": c.get("username", c.get("unique_id", "")),
                "language": c.get("language", ""),
                "classification": clf.get("classification", ""),
                "classification_reason": clf.get("classification_reason", "")
            }
            all_rows.append(row)
        time.sleep(2)

    if all_rows:
        df_comments = pd.DataFrame(all_rows)[COMMENT_COLS]
        append_to_sheet(service, SHEET_TT_DATA_COMMENTS_ID, TAB_TT_DATA_COMMENTS, df_comments)
        print(f"      {len(all_rows)} comentário(s) salvo(s) para vídeo {video_id}.", flush=True)

# ==============================
# MAIN
# ==============================

def main():
    print("=== TikTok Pipeline ===", flush=True)

    # Checar variáveis de ambiente
    print(f"SOCIAVAULT_API_KEY: {'OK' if SOCIAVAULT_API_KEY else 'FALTANDO'}", flush=True)
    print(f"GEMINI_API_KEY:     {'OK' if GEMINI_API_KEY else 'FALTANDO'}", flush=True)
    print(f"GDRIVE_CREDENTIALS: {'OK' if GDRIVE_CREDENTIALS else 'FALTANDO'}", flush=True)

    if not all([SOCIAVAULT_API_KEY, GEMINI_API_KEY, GDRIVE_CREDENTIALS]):
        print("ERRO: Variáveis de ambiente faltando. Abortando.", flush=True)
        return

    print("[INIT] Autenticando no Google Sheets...", flush=True)
    service = get_google_service()

    print("[INIT] Inicializando cliente Gemini...", flush=True)
    client = genai.Client(api_key=GEMINI_API_KEY)

    # ETAPA 1 — Ler perfis
    perfis = ler_perfis(service)
    if not perfis:
        return

    for perfil in perfis:
        username = perfil["profile"].lstrip("@")
        print(f"\n{'='*40}", flush=True)
        print(f"PERFIL: @{username}", flush=True)
        print(f"{'='*40}", flush=True)

        # ETAPA 2.0 — Dados do perfil
        try:
            processar_perfil(service, username)
        except Exception as e:
            print(f"  Erro em 2.0 para {username}: {e}. Pulando.", flush=True)
            continue

        # ETAPA 2.1 — Vídeos
        try:
            posts = processar_videos(service, username)
        except Exception as e:
            print(f"  Erro em 2.1 para {username}: {e}. Pulando.", flush=True)
            continue

        if not posts:
            print(f"  Sem posts para processar comentários de {username}.", flush=True)
            continue

        # ETAPA 2.2 — Comentários de cada vídeo
        for post in posts:
            try:
                processar_comentarios(service, client, post)
            except Exception as e:
                print(f"  Erro em 2.2 para vídeo {post.get('video_id', '?')}: {e}. Pulando.", flush=True)
                continue

    print("\n=== Pipeline finalizado ===", flush=True)


if __name__ == "__main__":
    main()
