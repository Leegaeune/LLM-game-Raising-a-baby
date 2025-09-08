import os
import json
import random
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, List, Tuple

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    st.error("⚠️ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    st.stop()



# 게임 데이터 -> 나이에 맞는 질문 설정해야함 
SITUATIONS = [

    {"age_range": [3, 3], "scenario": "마트에서 아이가 과자를 사달라며 바닥에 누워 떼를 쓰고 있습니다. 주변 사람들이 쳐다보고 있어요.", "context": "공공장소에서의 떼쓰기 상황"},

    {"age_range": [5, 5], "scenario": "유치원 친구가 갖고 싶었던 장난감을 갖고 놀고 있습니다. 아이가 떼를 쓰며 '나도 저거 갖고 싶어!'라고 소리칩니다.", "context": "나누기와 소유욕"},

    {"age_range": [7, 7], "scenario": "숙제를 하기 싫다며 책상 앞에 앉기만 하고 30분째 아무것도 하지 않고 있어요.", "context": "학습 습관과 책임감"},

    {"age_range": [9, 9], "scenario": "아이가 '나만 빼고 다 학원 다녀'라며 친구들과 비교하는 말을 합니다. 학원을 보내달라고 떼를 씁니다.", "context": "또래 압력과 교육열"},

    {"age_range": [11, 11], "scenario": "아이가 거짓말을 했다는 것을 알게 됐습니다. '숙제 다 했어'라고 했는데 실제로는 하지 않았어요.", "context": "정직함과 신뢰관계"},

    {"age_range": [13, 13], "scenario": "학교에서 시험 성적이 많이 떨어졌습니다. 아이는 '어차피 난 머리가 나쁘니까'라며 포기하는 모습을 보입니다.", "context": "자존감과 학습동기"},

    {"age_range": [15, 15], "scenario": "아이가 밤늦게까지 스마트폰을 보고 있다가 다음 날 학교에 늦었습니다. 지적하자 '친구들은 더 늦게까지 해'라고 반항합니다.", "context": "미디어 사용과 자율성"},

    {"age_range": [17, 17], "scenario": "아이가 늦은 시간에 들어와서 술냄새가 납니다. 친구들과 처음으로 술을 마셨다고 솔직하게 말합니다.", "context": "청소년기 일탈과 신뢰"}
]

INITIAL_GAME_STATE = {
    "child_age": 3,
    "child_name": "우리 아이",
    "happiness": 70,
    "growth": 50,
    "social": 60,
    "creativity": 55,
    "responsibility": 45,
    "total_situations": 0,
    "game_phase": "playing"
}

# OpenAI API 활용 분석 함수
import re
import json

def safe_json_parse(content: str):
    """응답 문자열에서 JSON만 추출해서 파싱"""
    try:
        # ```json 또는 ``` 제거
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        content = content.strip()

        # 숫자 앞의 + 기호 제거 (예: +5 -> 5)
        content = re.sub(r"\+(\d+)", r"\1", content)

        # 마지막 쉼표 제거 (예: {"a":1,} -> {"a":1})
        content = re.sub(r",\s*([}\]])", r"\1", content)

        # JSON 블록 찾기 (여러 개 중 첫 번째)
        matches = re.findall(r"\{[\s\S]*\}", content)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # 혹시 그냥 JSON일 수도 있으니 직접 시도
        return json.loads(content)
    except Exception:
        return None


def analyze_parenting_response(user_input: str, situation: dict, child_age: int) -> dict:
    """OpenAI API를 사용하여 부모의 대응을 분석하고 점수를 매기는 함수"""

    system_prompt = f"""
당신은 아동 발달 전문가이자 육아 상담사입니다.
부모가 {child_age}세 아이의 상황에 어떻게 대응했는지 분석하고 평가해주세요.

평가 기준:
- 행복도 (0-100): 아이의 즉각적인 감정과 만족감
- 성장지수 (0-100): 장기적인 발달과 학습
- 사회성 (0-100): 타인과의 관계 형성 능력
- 창의성 (0-100): 창의적 사고와 표현력
- 책임감 (0-100): 규칙 준수와 자율성

각 항목에 대해 -10에서 10 사이의 정수 점수 변화를 제안하고,
부모의 대응에 대한 따뜻하면서도 전문적인 피드백을 제공해주세요.

⚠️ 반드시 지켜야 할 규칙:
1. 출력은 **JSON 형식만** 포함하세요. 절대 텍스트나 설명을 붙이지 마세요.
2. 점수 변화는 **정수만 사용**, + 기호는 쓰지 마세요. 예: -5, 0, 3
3. JSON 코드블록(```json ... ```)이나 마크다운 문법을 사용하지 마세요.
4. 출력 형식은 반드시 아래와 같아야 합니다.

응답 형식:
{{
    "effects": {{
        "happiness": 점수변화(-10~10),
        "growth": 점수변화(-10~10),
        "social": 점수변화(-10~10),
        "creativity": 점수변화(-10~10),
        "responsibility": 점수변화(-10~10)
    }},
    "feedback": "부모 대응에 대한 피드백 (50자 이내)",
    "response_type": "대응 유형 (예: 공감형, 훈육형, 교육형 등)"
}}
"""

    user_prompt = f"""
상황: {situation['scenario']}
맥락: {situation['context']}
아이 나이: {child_age}세
부모의 대응: "{user_input}"

위 대응을 분석하고 평가해주세요.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()
        parsed = safe_json_parse(content)

        if not parsed:
            st.error("⚠️ 모델 응답에서 JSON을 추출하지 못했습니다. 응답: " + content)
            return {
                "effects": {"happiness": 0, "growth": 0, "social": 0, "creativity": 0, "responsibility": 0},
                "feedback": "모델 응답 형식 오류가 발생했습니다. 다시 시도해주세요.",
                "response_type": "오류"
            }

        return parsed

    except Exception as e:
        st.error(f"API 호출 오류가 발생했습니다: {e}")
        return {
            "effects": {"happiness": 0, "growth": 0, "social": 0, "creativity": 0, "responsibility": 0},
            "feedback": "API 오류로 분석할 수 없습니다. 잠시 후 다시 시도해주세요.",
            "response_type": "오류"
        }

# 게임 로직 함수
def get_situation_for_age(age: int) -> Tuple[Dict, int]:
    """나이에 맞는 상황을 선택하고, 사용하지 않은 상황을 우선적으로 선택합니다."""
    used_situations = st.session_state.used_situations
    available_indices = [i for i, sit in enumerate(SITUATIONS) if age >= sit["age_range"][0] and age <= sit["age_range"][1]]

    
    unused_available = list(set(available_indices) - set(used_situations))
    if unused_available:
        chosen_idx = random.choice(unused_available)
    else:
        
        chosen_idx = random.choice(available_indices) if available_indices else random.choice(list(range(len(SITUATIONS))))

    return SITUATIONS[chosen_idx], chosen_idx

def update_game_state(game_state: dict, effects: dict) -> dict:
    """게임 상태 업데이트 (0-100 범위 제한)"""
    new_state = game_state.copy()
    for key, change in effects.items():
        if key in new_state:
            new_state[key] = max(0, min(100, new_state[key] + change))
    return new_state

def get_final_result(game_state: dict) -> Dict[str, str]:
    """최종 결과 계산"""
    stats = game_state
    total = sum(stats[key] for key in ["happiness", "growth", "social", "creativity", "responsibility"])
    
    if total >= 400:
        return {"job": "훌륭한 리더", "description": "균형잡힌 성장을 한 멋진 어른이 되었어요! 🌟"}
    elif stats["responsibility"] >= 80:
        return {"job": "모범적인 공무원", "description": "책임감이 뛰어난 사회의 기둥이 되었네요! 🏛️"}
    elif stats["creativity"] >= 80:
        return {"job": "창의적인 예술가", "description": "창의력이 넘치는 예술가가 되었어요! 🎨"}
    elif stats["social"] >= 80:
        return {"job": "인기 많은 상담사", "description": "사람들과 잘 어울리는 따뜻한 어른이 되었네요! 🤝"}
    elif stats["growth"] >= 80:
        return {"job": "지적인 연구원", "description": "끊임없이 배우고 성장하는 학자가 되었어요! 📚"}
    elif stats["happiness"] >= 80:
        return {"job": "행복한 자유인", "description": "긍정적이고 밝은 에너지를 가진 사람이 되었네요! ☀️"}
    elif total < 200:
        return {"job": "방황하는 청년", "description": "아직 자신의 길을 찾아가는 중이에요. 더 많은 관심이 필요했을지도... 😔"}
    else:
        return {"job": "평범한 직장인", "description": "무난하게 성장한 평범하지만 소중한 어른이 되었어요! 😊"}


# Streamlit UI 구성 및 게임 실행
def reset_game():
    """게임 초기화"""
    st.session_state.game_state = INITIAL_GAME_STATE.copy()
    st.session_state.game_history = []
    st.session_state.used_situations = []
    st.session_state.current_situation = None
    st.session_state.current_situation_idx = None

def process_user_response(user_input: str):
    """사용자 응답 처리 및 게임 상태 업데이트"""
    with st.spinner("AI가 당신의 대응을 분석하고 있습니다..."):
        analysis = analyze_parenting_response(user_input, st.session_state.current_situation, st.session_state.game_state["child_age"])

    if analysis["response_type"] != "오류":
        st.session_state.game_state = update_game_state(st.session_state.game_state, analysis["effects"])
        
        st.session_state.game_history.append({
            "situation": st.session_state.current_situation,
            "user_response": user_input,
            "analysis": analysis,
            "age": st.session_state.game_state["child_age"]
        })
        st.session_state.used_situations.append(st.session_state.current_situation_idx)
        
        
        st.session_state.game_state["child_age"] += 2
        
        st.session_state.game_state["total_situations"] += 1

    if st.session_state.game_state["total_situations"] >= 8:
        st.session_state.game_state["game_phase"] = "result"
    else:
        situation, idx = get_situation_for_age(st.session_state.game_state["child_age"])
        st.session_state.current_situation = situation
        st.session_state.current_situation_idx = idx
    st.rerun()

def display_game_area():
    """게임 메인 영역 표시"""
    
    if st.session_state.current_situation is None:
        st.info("게임 데이터를 불러오는 중입니다. 잠시만 기다려주세요...")
        return

    current_situation = st.session_state.current_situation
    st.subheader("🎯 현재 상황")
    with st.container(border=True):
        st.info(f"**{current_situation['context']}**")
        st.write(current_situation['scenario'])

    st.subheader("💭 어떻게 대응하시겠어요?")
    user_input = st.text_area(
        "부모로서 어떻게 대응할지 자유롭게 적어주세요:",
        placeholder="예: 아이를 조용한 곳으로 데려가서 진정할 때까지 기다린 후, 왜 화가 났는지 물어본다.",
        height=120,
        key=f"user_response_{st.session_state.game_state['total_situations']}"
    )

    if st.button("🚀 대응하기", type="primary", disabled=not user_input.strip(), key="respond_button"):
        process_user_response(user_input)
    if st.button("🔄 게임 초기화", use_container_width=True, key="reset_button"):
        reset_game()

    if st.session_state.game_history:
        st.markdown("---")
        st.subheader("📝 최근 결과")
        latest_result = st.session_state.game_history[-1]
        with st.expander("결과 보기", expanded=True):
            st.write(f"**당신의 대응:** `{latest_result['user_response']}`")
            st.write(f"**분석 결과:** {latest_result['analysis']['feedback']}")
            st.write(f"**대응 유형:** {latest_result['analysis']['response_type']}")
            
            st.markdown("##### 📈 능력치 변화")
            effects = latest_result['analysis']['effects']
            cols = st.columns(len(effects))
            for i, (key, value) in enumerate(effects.items()):
                with cols[i]:
                    emoji = {"happiness": "❤️", "growth": "📚", "social": "👥", "creativity": "💡", "responsibility": "⚖️"}
                    sign = "+" if value >= 0 else ""
                    st.metric(f"{emoji.get(key, '')} {key.capitalize()}", f"{sign}{value}")
                    
def display_result_screen():
    """최종 결과 화면"""
    st.success("🎉 성장 완료!")
    result = get_final_result(st.session_state.game_state)
    st.title(result['job'])
    st.write(result['description'])
    
    st.markdown("---")
    st.subheader("📊 최종 능력치")
    stats = st.session_state.game_state
    stat_names = ["happiness", "growth", "social", "creativity", "responsibility"]
    stat_data = {
        "능력치": [s.capitalize() for s in stat_names],
        "점수": [stats[s] for s in stat_names]
    }
    
    import pandas as pd
    df = pd.DataFrame(stat_data)
    st.bar_chart(df.set_index('능력치'))

    st.button("🔄 다시 키워보기", type="primary", on_click=reset_game)

def display_stats_sidebar():
    """사이드바 능력치 표시"""
    st.sidebar.subheader("📊 아이 상태")
    stats = st.session_state.game_state
    
    st.sidebar.markdown("---")
    st.sidebar.progress(stats["happiness"] / 100, text=f"❤️ 행복도: {stats['happiness']}/100")
    st.sidebar.progress(stats["growth"] / 100, text=f"📚 성장지수: {stats['growth']}/100")
    st.sidebar.progress(stats["social"] / 100, text=f"👥 사회성: {stats['social']}/100")
    st.sidebar.progress(stats["creativity"] / 100, text=f"💡 창의성: {stats['creativity']}/100")
    st.sidebar.progress(stats["responsibility"] / 100, text=f"⚖️ 책임감: {stats['responsibility']}/100")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 게임 정보")
    st.sidebar.info("""
    - 총 8가지 상황을 경험합니다.
    - 자유롭게 대응 방법을 입력하세요.
    - AI가 당신의 육아 방식을 분석합니다.
    - 최종 결과로 아이가 어떤 어른이 될지 확인하세요!
    """)
    

def main():
    """Streamlit 앱 메인 함수"""
    st.set_page_config(page_title="애기키우기 시뮬레이션", page_icon="👶", layout="wide")
    st.title("🍼 애기키우기 시뮬레이션")
    st.markdown("---")

    if "game_state" not in st.session_state:
        reset_game()
        
        situation, idx = get_situation_for_age(st.session_state.game_state["child_age"])
        st.session_state.current_situation = situation
        st.session_state.current_situation_idx = idx

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("현재 나이", f"{st.session_state.game_state['child_age']}세")
    with col2:
        st.metric("경험한 상황", f"{st.session_state.game_state['total_situations']}/8")
    with col3:
        st.metric("게임 단계", "진행 중" if st.session_state.game_state["game_phase"] == "playing" else "완료")
    
    st.markdown("---")

    
    col_main, col_sidebar = st.columns([2, 1])

    with col_main:
        if st.session_state.game_state["game_phase"] == "playing":
            display_game_area()
        else:
            display_result_screen()

    with col_sidebar:
        display_stats_sidebar()

if __name__ == "__main__":
    main()



# 청소년기 (10~18세경): 자아정체감 확립 
# 
