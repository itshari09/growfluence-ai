import os
import json
import uuid
import html
from datetime import date, datetime, timedelta
from typing import Any

import requests
import streamlit as st

st.set_page_config(page_title="Growfluence AI", page_icon="✳", layout="wide", initial_sidebar_state="expanded")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { color-scheme: dark; }
.stApp { background: #090b10; color: #f4f5f7; font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: #0e1119; border-right: 1px solid #202532; }
h1,h2,h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }
.hero { padding: 28px 30px; border: 1px solid #272d3a; border-radius: 22px; background: radial-gradient(circle at 85% 20%, #273e35 0, transparent 34%), linear-gradient(135deg,#151b25,#10141c); margin-bottom: 22px; }
.eyebrow { color:#8de2b0; font-size:12px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
.hero h1 { font-size: clamp(32px,5vw,52px); margin: 10px 0 8px; }
.muted { color:#9ba4b4; }
.metric { background:#121722; border:1px solid #252d3b; border-radius:16px; padding:18px; min-height:105px; }
.metric-label { color:#9ba4b4; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { color:#f4f5f7; font-size:27px; font-weight:800; margin-top:8px; font-family:Manrope,sans-serif; }
.panel { background:#111620; border:1px solid #252d3b; border-radius:18px; padding:20px; margin-bottom:14px; }
.pill { display:inline-block; background:#1b3028; color:#9be8b9; padding:5px 10px; border-radius:20px; font-size:11px; font-weight:700; }
div.stButton > button { border-radius:11px; min-height:42px; font-weight:700; }
div[data-testid="stForm"] { border:1px solid #252d3b; border-radius:16px; padding:18px; }
hr { border-color:#252d3b; }
</style>
""", unsafe_allow_html=True)

def api_headers(token: str | None = None):
    h = {"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def auth_request(path: str, payload: dict):
    response = requests.post(f"{SUPABASE_URL}/auth/v1/{path}", headers=api_headers(), json=payload, timeout=20)
    if response.status_code >= 400:
        try: message = response.json().get("msg") or response.json().get("message") or response.text
        except Exception: message = response.text
        raise RuntimeError(message)
    return response.json()

def db_request(table: str, method: str = "GET", payload: Any = None, params: dict | None = None):
    token = st.session_state.auth.get("access_token")
    response = requests.request(method, f"{SUPABASE_URL}/rest/v1/{table}", headers={**api_headers(token), "Prefer":"return=representation"}, json=payload, params=params, timeout=25)
    if response.status_code >= 400:
        raise RuntimeError(response.text[:500])
    if response.status_code == 204 or not response.text:
        return []
    return response.json()

def signed_in():
    return bool(st.session_state.get("auth", {}).get("access_token"))

def ai_json(system_prompt: str, user_prompt: str):
    if not OPENAI_API_KEY:
        return None
    response = requests.post("https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"},
        json={"model":OPENAI_MODEL,"temperature":0.75,"response_format":{"type":"json_object"},
              "messages":[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]}, timeout=60)
    response.raise_for_status()
    return json.loads(response.json()["choices"][0]["message"]["content"])

def profile():
    rows = db_request("creator_profiles", params={"select":"*","limit":"1"})
    return rows[0] if rows else None

def save_profile(data: dict):
    data = {**data, "user_id": st.session_state.auth.get("user", {}).get("id")}
    existing = profile()
    if existing:
        db_request("creator_profiles", "PATCH", data, {"id":f"eq.{existing['id']}"})
    else:
        db_request("creator_profiles", "POST", data)

def fallback_week(niche: str, audience: str, platforms: list[str], goal: str, count: int = 7):
    formats = ["Short-form video", "Carousel", "Story", "Educational post", "Behind-the-scenes", "Short-form video", "Community post"]
    angles = [
        f"The biggest mistake people make in {niche}", f"5 practical {niche} tips beginners can use",
        f"Ask your {audience} audience a question about {niche}", f"A step-by-step {niche} tutorial",
        f"Behind the scenes: how you approach {niche}", f"An unpopular opinion about {niche}",
        f"Share a lesson and invite your community to respond"
    ]
    return [{"day":(date.today()+timedelta(days=i)).isoformat(),"platform":platforms[i % len(platforms)] if platforms else "Instagram",
             "format":formats[i],"topic":angles[i],"hook":f"Stop scrolling if you want to improve your {niche}.",
             "caption":f"Let’s make {niche} easier. Here is one idea you can put into action today. Save this for later and tell me what you want to learn next.",
             "cta":"Save this post and share your experience in the comments.","hashtags":[f"#{niche.replace(' ','')}", "#creatorcommunity", "#learnandgrow"],
             "objective":goal,"status":"Idea"} for i in range(count)]

def make_week(p: dict):
    count = max(3, min(14, int(p.get('posts_per_week', 7))))
    prompt = f"""Create a realistic {count}-post content plan for a creator. Return JSON with key posts, an array of exactly {count} objects. Each object: day (YYYY-MM-DD), platform, format, topic, hook, caption, cta, hashtags (array of 3-6), objective, status (Idea). Use distinct ideas, a sensible mix of formats, platform-aware copy, and do not promise viral results. Start {date.today().isoformat()}."""
    try:
        result = ai_json("You are a senior creator growth strategist. Return valid concise JSON only.", json.dumps({"creator_profile":p,"request":prompt}))
        posts = result.get("posts", [])
        if len(posts) == count: return posts, "AI-generated plan"
    except Exception as e:
        st.warning(f"AI planner unavailable; using a practical starter plan. Details: {str(e)[:180]}")
    return fallback_week(p.get("niche","your niche"), p.get("target_audience","your audience"), p.get("platforms",["Instagram"]), p.get("primary_goal","Engagement"), count), "Starter plan"

def get_posts():
    return db_request("content_posts", params={"select":"*","order":"scheduled_for.asc","limit":"100"})

def page_header(kicker: str, title: str, subtitle: str):
    st.markdown(f'<div class="eyebrow">{kicker}</div><h1>{title}</h1><p class="muted">{subtitle}</p>', unsafe_allow_html=True)

if "auth" not in st.session_state: st.session_state.auth = {}
if "active_page" not in st.session_state: st.session_state.active_page = "Overview"

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.markdown('<div class="hero"><div class="eyebrow">CREATOR GROWTH WORKSPACE</div><h1>Growfluence <span style="color:#8de2b0">AI</span></h1><p class="muted">Your strategy, content calendar and growth insights — in one focused workspace.</p></div>', unsafe_allow_html=True)
    st.error("Supabase is not configured yet. Add SUPABASE_URL and SUPABASE_ANON_KEY to your deployment environment to enable secure multi-user accounts.")
    st.code("SUPABASE_URL=https://YOUR_PROJECT.supabase.co\nSUPABASE_ANON_KEY=your_supabase_anon_key\nOPENAI_API_KEY=your_openai_api_key")
    st.stop()

if not signed_in():
    st.markdown('<div class="hero"><div class="eyebrow">THE CREATOR GROWTH STUDIO</div><h1>Make every post<br>move you forward.</h1><p class="muted">Plan with purpose. Publish consistently. Learn what works.</p></div>', unsafe_allow_html=True)
    login_tab, signup_tab = st.tabs(["Sign in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            try:
                st.session_state.auth = auth_request("token?grant_type=password", {"email":email,"password":password})
                st.rerun()
            except Exception as e: st.error(f"Sign in failed: {e}")
    with signup_tab:
        with st.form("signup_form"):
            name = st.text_input("Your name")
            email = st.text_input("Email address", key="signup_email")
            password = st.text_input("Password (at least 8 characters)", type="password", key="signup_password")
            agree = st.checkbox("I agree to the terms and privacy policy.")
            submitted = st.form_submit_button("Create free account", use_container_width=True)
        if submitted:
            if not agree: st.error("Please accept the terms to continue.")
            elif len(password) < 8: st.error("Use a password with at least 8 characters.")
            else:
                try:
                    result = auth_request("signup", {"email":email,"password":password,"data":{"full_name":name}})
                    if result.get("access_token"):
                        st.session_state.auth = result
                        st.rerun()
                    else: st.success("Account created. Check your email to confirm it, then sign in.")
                except Exception as e: st.error(f"Could not create account: {e}")
    st.caption("Your workspace is private to your account. Connect social accounts only through official platform authorization when available.")
    st.stop()

with st.sidebar:
    st.markdown("## ✳ Growfluence")
    st.caption("CREATOR GROWTH STUDIO")
    st.divider()
    pages = ["Overview","Weekly planner","Content studio","My strategy","Analytics"]
    selected = st.radio("WORKSPACE", pages, label_visibility="collapsed", key="navigation")
    st.divider()
    st.caption(f"Signed in as {st.session_state.auth.get('user',{}).get('email','creator')}")
    if st.button("Sign out", use_container_width=True):
        st.session_state.auth = {}
        st.rerun()

try:
    creator = profile()
    if selected == "My strategy":
        page_header("YOUR FOUNDATION","Creator strategy","Tell Growfluence who you create for and what you want to achieve.")
        with st.form("profile_form"):
            name = st.text_input("Creator / brand name", value=(creator or {}).get("display_name",""))
            niche = st.text_input("Your niche", value=(creator or {}).get("niche",""))
            audience = st.text_input("Who is your target audience?", value=(creator or {}).get("target_audience",""))
            platforms = st.multiselect("Platforms", ["Instagram","YouTube","TikTok","LinkedIn","X","Facebook","Pinterest"], default=(creator or {}).get("platforms",["Instagram"]))
            goal = st.selectbox("Primary growth goal", ["Grow audience","Increase engagement","Generate leads","Build authority","Sell products or services"], index=["Grow audience","Increase engagement","Generate leads","Build authority","Sell products or services"].index((creator or {}).get("primary_goal","Grow audience")) if (creator or {}).get("primary_goal","Grow audience") in ["Grow audience","Increase engagement","Generate leads","Build authority","Sell products or services"] else 0)
            voice = st.text_input("Your content voice", value=(creator or {}).get("brand_voice","Friendly, useful and authentic"))
            frequency = st.slider("Posts per week", 3, 14, int((creator or {}).get("posts_per_week",7)))
            saved = st.form_submit_button("Save my strategy", use_container_width=True)
        if saved:
            save_profile({"display_name":name,"niche":niche,"target_audience":audience,"platforms":platforms,"primary_goal":goal,"brand_voice":voice,"posts_per_week":frequency})
            st.success("Your creator strategy has been saved.")
            st.rerun()
        if creator: st.info("Your strategy is used to personalize weekly plans and content suggestions.")
    elif selected == "Weekly planner":
        page_header("PLAN WITH PURPOSE","Your weekly content plan","Build a balanced week of posts around your audience and goals.")
        if not creator or not creator.get("niche"):
            st.warning("Set up your creator strategy first.")
            st.caption("Choose My strategy from the sidebar to get started.")
        else:
            if st.button(f"✳ Generate my {int(creator.get('posts_per_week', 7))}-post plan", type="primary", use_container_width=True):
                items, source = make_week(creator)
                records = [{"scheduled_for":x.get("day",date.today().isoformat()),"platform":x.get("platform","Instagram"),"format":x.get("format","Post"),"topic":x.get("topic",""),"hook":x.get("hook",""),"caption":x.get("caption",""),"cta":x.get("cta",""),"hashtags":x.get("hashtags",[]),"objective":x.get("objective",creator.get("primary_goal","Grow audience")),"status":"Idea"} for x in items]
                db_request("content_posts","POST",records)
                st.success(f"{source} saved to your calendar.")
                st.rerun()
            posts = get_posts()
            if not posts: st.info("Your calendar is empty. Generate your first weekly plan.")
            for item in posts:
                with st.expander(f"{item.get('scheduled_for','')} · {item.get('platform','')} · {item.get('format','')} — {item.get('topic','')}"):
                    st.write(f"**Hook:** {item.get('hook','')}")
                    st.write(item.get("caption",""))
                    st.write(" ".join(item.get("hashtags") or []))
                    col1,col2=st.columns([2,1])
                    with col1:
                        new_status=st.selectbox("Status",["Idea","Draft","Ready","Published"],index=["Idea","Draft","Ready","Published"].index(item.get("status","Idea")),key=f"status_{item['id']}")
                    with col2:
                        if st.button("Save status",key=f"save_{item['id']}"):
                            db_request("content_posts","PATCH",{"status":new_status},{"id":f"eq.{item['id']}"})
                            st.rerun()
    elif selected == "Content studio":
        page_header("CREATE BETTER CONTENT","AI content studio","Turn one idea into a strong, platform-ready post.")
        if not creator: st.warning("Set up your strategy first.")
        else:
            topic=st.text_area("What do you want to post about?",placeholder="e.g. 3 mistakes new creators make")
            platform=st.selectbox("Platform",creator.get("platforms") or ["Instagram"])
            fmt=st.selectbox("Format",["Instagram Reel","Carousel","YouTube Short","YouTube video","LinkedIn post","Story","X thread"])
            tone=st.selectbox("Tone",["Friendly and helpful","Bold and opinionated","Educational","Personal storytelling","Professional"])
            if st.button("Generate content",type="primary"):
                prompt={"niche":creator.get("niche"),"audience":creator.get("target_audience"),"brand_voice":tone,"platform":platform,"format":fmt,"topic":topic}
                try:
                    result=ai_json("You are an expert social content editor. Return JSON with keys hook, caption, script, cta, hashtags (array), creative_direction. Keep it specific, useful, natural and platform-appropriate.",json.dumps(prompt))
                except Exception as e:
                    result=None; st.warning(f"AI provider unavailable: {e}")
                if result:
                    st.session_state.generated_content=result
                else:
                    st.session_state.generated_content={"hook":f"Most creators overlook this about {topic}.","caption":f"If you're interested in {creator.get('niche')}, here's one practical thought about {topic}. Make it useful, keep it honest, and invite your audience into the conversation.","script":"Hook (0–3s): State the problem.\nValue (3–20s): Share three useful points.\nClose: Give one clear next step.","cta":"Save this and share your take below.","hashtags":["#creatorgrowth","#contentstrategy"],"creative_direction":"Use a clean opening frame, readable captions and one clear visual idea."}
            if st.session_state.get("generated_content"):
                g=st.session_state.generated_content
                st.markdown('<div class="panel">',unsafe_allow_html=True)
                st.subheader(g.get("hook",""))
                st.write(g.get("caption",""))
                st.write(g.get("script",""))
                st.write(" ".join(g.get("hashtags",[])))
                st.caption("CTA: "+g.get("cta",""))
                st.caption("Creative direction: "+g.get("creative_direction",""))
                st.markdown('</div>',unsafe_allow_html=True)
                st.download_button("Download content brief",json.dumps(g,indent=2),"growfluence-content-brief.json","application/json")
    elif selected == "Analytics":
        page_header("MEASURE WHAT MATTERS","Growth analytics","Track your progress with creator-reported metrics.")
        posts=get_posts()
        published=[p for p in posts if p.get("status")=="Published"]
        st.caption("Metrics are manually entered until official social platform connections are configured.")
        with st.form("metrics_form"):
            metric_date=st.date_input("Date",value=date.today())
            platform=st.selectbox("Platform",["Instagram","YouTube","TikTok","LinkedIn","X","Facebook"])
            followers=st.number_input("Followers",min_value=0,step=1)
            reach=st.number_input("Reach",min_value=0,step=1)
            engagement=st.number_input("Engagements (likes + comments + shares + saves)",min_value=0,step=1)
            saves=st.number_input("Saves",min_value=0,step=1)
            add=st.form_submit_button("Save metrics")
        if add:
            db_request("creator_metrics","POST",{"recorded_on":metric_date.isoformat(),"platform":platform,"followers":followers,"reach":reach,"engagements":engagement,"saves":saves})
            st.success("Metrics saved.")
            st.rerun()
        metrics=db_request("creator_metrics",params={"select":"*","order":"recorded_on.asc","limit":"120"})
        if metrics:
            import pandas as pd
            df=pd.DataFrame(metrics)
            a,b,c=st.columns(3)
            a.metric("Latest followers",f"{int(df.sort_values('recorded_on').iloc[-1]['followers']):,}")
            b.metric("Total reach logged",f"{int(df['reach'].sum()):,}")
            rate=(df["engagements"].sum()/df["reach"].sum()*100) if df["reach"].sum() else 0
            c.metric("Engagement / reach",f"{rate:.1f}%")
            st.line_chart(df.pivot_table(index="recorded_on",columns="platform",values="followers",aggfunc="max").fillna(0))
            st.bar_chart(df.groupby("platform")[["reach","engagements"]].sum())
        else: st.info("Add your first weekly metrics snapshot to start seeing trends.")
    else:
        page_header("YOUR CREATOR COMMAND CENTRE","Good to see you again.","A simple view of what you're creating and where you're heading.")
        if not creator or not creator.get("niche"):
            st.markdown('<div class="panel"><h3>Start with your creator strategy</h3><p class="muted">Add your niche, audience and goals. Your weekly planner will use these details to create a focused plan.</p></div>',unsafe_allow_html=True)
            st.caption("Choose My strategy from the sidebar to get started.")
        else:
            safe_name = html.escape(str(creator.get('display_name') or 'Your creator brand'))
            safe_niche = html.escape(str(creator.get('niche') or ''))
            safe_goal = html.escape(str(creator.get('primary_goal') or ''))
            safe_platforms = html.escape(', '.join(creator.get('platforms') or []))
            st.markdown(f'<div class="hero"><div class="eyebrow">YOUR CREATOR SNAPSHOT</div><h1>{safe_name}</h1><p class="muted">{safe_niche} · {safe_goal} · {safe_platforms}</p></div>',unsafe_allow_html=True)
            posts=get_posts()
            week_start=date.today().isoformat()
            week_end=(date.today()+timedelta(days=6)).isoformat()
            this_week=[p for p in posts if week_start<=p.get("scheduled_for","")<=week_end]
            a,b,c,d=st.columns(4)
            for col,label,value in [(a,"Planned this week",len(this_week)),(b,"Ready to publish",sum(p.get("status")=="Ready" for p in posts)),(c,"Published",sum(p.get("status")=="Published" for p in posts)),(d,"Platforms",len(creator.get("platforms") or []))]:
                col.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)
            st.subheader("This week's focus")
            if this_week:
                for p in this_week[:5]:
                    safe_status = html.escape(str(p.get('status', 'Idea')))
                    safe_date = html.escape(str(p.get('scheduled_for', '')))
                    safe_platform = html.escape(str(p.get('platform', '')))
                    safe_topic = html.escape(str(p.get('topic', '')))
                    safe_format = html.escape(str(p.get('format', '')))
                    st.markdown(f'<div class="panel"><span class="pill">{safe_status}</span> &nbsp; <b>{safe_date}</b> · {safe_platform}<h3>{safe_topic}</h3><span class="muted">{safe_format}</span></div>',unsafe_allow_html=True)
            else: st.info("Your week is open. Head to Weekly planner to create your first content schedule.")
            st.caption("Open Weekly planner from the sidebar to create or update your schedule.")
except requests.RequestException as e:
    st.error(f"Could not reach the service. Check your Supabase URL and network connection. {e}")
except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.caption("If this persists, check the Supabase schema and environment variables in the setup guide.")
