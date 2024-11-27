from langchain_openai import ChatOpenAI
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, Form, Depends
from fastapi.responses import JSONResponse
from typing import Dict
import openai
import os
import json

app = FastAPI()

# 각 챌린지에 대한 클래스 이름 설정
class_names = {
    "transport": ["a-station-nameplate"],
    "recycling": ["plastic-bottle"],
    "tumbler": ["tumblr"]
}

def classify_image_yolo(model_path, image_path: str) -> str:
    """
    YOLO 모델을 사용하여 이미지에서 객체를 분류하고 신뢰도가 0.65 이상일 경우 해당 클래스를 반환합니다.
    """
    model = YOLO(model_path)
    result = model(image_path)

    if len(result) > 0:
        for res in result:
            boxes = res.boxes
            if boxes is not None and len(boxes.conf) > 0:
                conf = boxes.conf[0].item()  # 첫 번째 박스의 신뢰도 값을 가져옵니다.
                print("신뢰도(conf):", conf)

                if conf >= 0.65:
                    # 신뢰도가 0.65 이상인 경우 클래스 반환
                    class_id = int(boxes.cls[0])
                    return model.names[class_id]  # 클래스 이름 반환
                else:
                    return "low_confidence"
    return "unknown"

# 파일 저장 경로 설정
UPLOAD_DIRECTORY = "uploaded_files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# 대중교통 챌린지 엔드포인트
@app.post("/transport-challenge")
async def transport_challenge(file: UploadFile = File(...)):
    model_path = r"C:\sul_projects\python_basic\12_ngrok\yolo\runs\detect\train27\weights\best.pt"
    """
    대중교통 챌린지: 버스정류장 또는 지하철역 사진인지 확인합니다.
    """
    try:
        # 파일 저장
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # 이미지 경로로 클래스 분류
        class_label = classify_image_yolo(model_path, file_path)

        # 임시로 저장한 파일 삭제
        os.remove(file_path)

        if class_label == "low_confidence":
            return JSONResponse(content={"message": "인증 불가!"})
        elif class_label in class_names["transport"]:
            return JSONResponse(content={"message": "인증 완료!"})
        else:
            return JSONResponse(content={"message": "인증 불가!"})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 쓰레기 분리수거 챌린지 엔드포인트
@app.post("/recycling-challenge")
async def recycling_challenge(file: UploadFile = File(...)):
    model_path = r"C:\sul_projects\python_basic\12_ngrok\yolo\runs\detect\train28\weights\best.pt"
    """
    쓰레기 분리수거 챌린지: 빈 페트병 사진인지 확인합니다.
    """
    try:
        # 파일 저장
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # 이미지 경로로 클래스 분류
        class_label = classify_image_yolo(model_path, file_path)

        # 임시로 저장한 파일 삭제
        os.remove(file_path)

        if class_label == "low_confidence":
            return JSONResponse(content={"message": "인증 불가!"})
        elif class_label in class_names["recycling"]:
            return JSONResponse(content={"message": "인증 완료!"})
        else:
            return JSONResponse(content={"message": "인증 불가!"})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 텀블러 이용 챌린지 엔드포인트
@app.post("/tumbler-challenge")
async def tumbler_challenge(file: UploadFile = File(...)):
    model_path = r"C:\sul_projects\python_basic\12_ngrok\yolo\runs\detect\train15\weights\best.pt"
    """
    텀블러 이용 챌린지: 텀블러 사진인지 확인합니다.
    """
    try:
        # 파일 저장
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:   
            buffer.write(await file.read())

        # 이미지 경로로 클래스 분류
        class_label = classify_image_yolo(model_path, file_path)

        # 임시로 저장한 파일 삭제
        os.remove(file_path)

        if class_label == "low_confidence":
            return JSONResponse(content={"message": "인증 불가!"})
        elif class_label in class_names["tumbler"]:
            return JSONResponse(content={"message": "인증 완료!"})
        else:
            return JSONResponse(content={"message": "인증 불가!"})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# OpenAI Chatbot 설정

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key  # OpenAI API 키 설정

class OpenAIChat:
    def __init__(self, model_name='gpt-4o-mini'):
        self.model_name = model_name

    def chat(self, message: str, persona: str, history: list):
        # OpenAI API 호출 (동기 처리)
        messages = [{"role": "system", "content": persona}] + history + [{"role": "user", "content": message}]
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages
        )
        return completion.choices[0].message['content']
    
    def load_persona_from_file(self, file_path: str):
        # 파일에서 페르소나 읽기 (UTF-8 인코딩 사용)
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()


# 세션 관리 (사용자별 대화 기록 및 퀘스트 상태)
user_sessions = {}

# 파일에 대화 기록 저장
def save_conversation(user_id: str):
    with open(f"conversation_{user_id}.json", "w", encoding="utf-8") as file:
        json.dump(user_sessions[user_id], file, ensure_ascii=False, indent=4)

# 파일에서 대화 기록 불러오기
def load_conversation(user_id: str):
    try:
        with open(f"conversation_{user_id}.json", "r", encoding="utf-8") as file:
            user_sessions[user_id] = json.load(file)
    except FileNotFoundError:
        user_sessions[user_id] = {"history": [], "initialized": False, "quest": False, "quest_completed": False}

# OpenAI Chatbot 설정
class OpenAIChat:
    def __init__(self, model_name='gpt-4o-mini'):
        self.model_name = model_name

    def chat(self, message: str, persona: str, history: list):
        messages = [{"role": "system", "content": persona}] + history + [{"role": "user", "content": message}]
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages
        )
        return completion.choices[0].message['content']

    def load_persona_from_file(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

chat = OpenAIChat()

animals = {
    "dolphin": {
        "name": "돌고래",
        "intro": "안녕~~!! 나는 돌고래야~~!! 겁이 없고 사람을 좋아해서 자주 웃어~ 궁금한 거 다 물어봐도 좋아~!",
        "persona_file": r"C:\sul_projects\python_basic\12_ngrok\돌고래.txt"
    },
    "seahorse": {
        "name": "해마",
        "intro": "안녕 난 해마야. 사실 의외로 먹보야! 그래서 궁금한 게 뭐라고?! ",
        "persona_file": r"C:\sul_projects\python_basic\12_ngrok\해마.txt"
    },
    "clownfish": {
        "name": "흰동가리",
        "intro": "안녕! 나는 흰동가리야ㅎㅎ 뭐가 궁금해?",
        "persona_file": r"C:\sul_projects\python_basic\12_ngrok\흰동가리.txt"
    },
    "turtle": {
        "name": "거북이", 
        "intro": "안녕하시게... 나는 거북라네... 느리고 인자한 성격으로 독립적인 생활이 익숙하지... 궁금한 게 있다면 뭐든 알려주겠네",
        "persona_file": r"C:\sul_projects\python_basic\12_ngrok\거북이.txt"
    }
}




def handle_help_request(user_id: str, animal_name: str, persona: str):
    success = True  # 임의로 성공 처리
    message = "퀘스트 성공!" if success else "퀘스트 실패!"
    return chat.chat(message, persona, user_sessions[user_id]["history"])


# 돌고래와의 대화
@app.post("/chat/dolphin")
async def dolphin_chat(user_id: str, user_message: str = Form(...), trash_count: int = Form(0)):
    try:
        load_conversation(user_id)

        # 첫 대화 시 인트로 메시지 반환
        if user_id not in user_sessions:
            user_sessions[user_id] = {"history": [], "initialized": True, "quest": False}
            animal_intro = animals["dolphin"]["intro"]
            return JSONResponse(content={"message": animal_intro})

        # 퀘스트가 진행 중일 경우
        if user_sessions[user_id]["quest"]:
            if trash_count >= 5:
                user_sessions[user_id]["quest_completed"] = True
                user_sessions[user_id]["quest"] = False

                # OpenAI를 사용해 랜덤한 퀘스트 완료 메시지 생성
                persona_file_path = animals["dolphin"]["persona_file"]
                persona = chat.load_persona_from_file(persona_file_path)

                completion_message = "우와! 바다가 정말 깨끗해졌어! 정말 고마워!"
                response = chat.chat(completion_message, persona, user_sessions[user_id]["history"])

                # history에 추가
                user_sessions[user_id]["history"].append({"role": "assistant", "content": response})
                save_conversation(user_id)

                return JSONResponse(content={"message": response})
            else:
                # 쓰레기가 아직 남아있을 경우 랜덤한 메시지 생성
                persona_file_path = animals["dolphin"]["persona_file"]
                persona = chat.load_persona_from_file(persona_file_path)

                reminder_message = "아직 조금 더 쓰레기를 치워줘! 조금만 더 힘내면 돼!"
                response = chat.chat(reminder_message, persona, user_sessions[user_id]["history"])

                # history에 추가
                user_sessions[user_id]["history"].append({"role": "assistant", "content": response})
                save_conversation(user_id)

                return JSONResponse(content={"message": response})

        # 사용자가 도움 요청 시 퀘스트 시작
        if "도움" in user_message:
            user_sessions[user_id]["quest"] = True
            save_conversation(user_id)
            return JSONResponse(content={"message": "좋아! 바닷속 쓰레기를 좀 주워줄래?"})

        # 기본 대화 처리
        persona_file_path = animals["dolphin"]["persona_file"]
        persona = chat.load_persona_from_file(persona_file_path)

        user_sessions[user_id]["history"].append({"role": "user", "content": user_message})
        response = chat.chat(user_message, persona, user_sessions[user_id]["history"])
        user_sessions[user_id]["history"].append({"role": "assistant", "content": response})

        save_conversation(user_id)

        return JSONResponse(content={"message": response})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)




# 해마와의 대화
@app.post("/chat/seahorse")
async def seahorse_chat(user_id: str, user_message: str = Form(...), trash_count: int = Form(0)):
    try:
        load_conversation(user_id)

        # 첫 대화 시 인트로 메시지 반환
        if user_id not in user_sessions:
            user_sessions[user_id] = {"history": [], "initialized": True, "quest": False}
            animal_intro = animals["seahorse"]["intro"]
            return JSONResponse(content={"message": animal_intro})

        # 퀘스트가 진행 중일 경우
        if user_sessions[user_id]["quest"]:
            if trash_count >= 5:
                user_sessions[user_id]["quest_completed"] = True
                user_sessions[user_id]["quest"] = False

                # OpenAI를 사용해 랜덤한 퀘스트 완료 메시지 생성
                persona_file_path = animals["seahorse"]["persona_file"]
                persona = chat.load_persona_from_file(persona_file_path)

                completion_message = "고마워! 산호 주변이 깨끗해졌어!"
                response = chat.chat(completion_message, persona, user_sessions[user_id]["history"])

                # history에 추가
                user_sessions[user_id]["history"].append({"role": "assistant", "content": response})
                save_conversation(user_id)

                return JSONResponse(content={"message": response})
            else:
                # 쓰레기가 아직 남아있을 경우 랜덤한 메시지 생성
                persona_file_path = animals["seahorse"]["persona_file"]
                persona = chat.load_persona_from_file(persona_file_path)

                reminder_message = "아직 퀘스트가 남아있어. 쓰레기가 좀 더 남아있나봐. 조금만 더 힘내!"
                response = chat.chat(reminder_message, persona, user_sessions[user_id]["history"])

                # history에 추가
                user_sessions[user_id]["history"].append({"role": "assistant", "content": response})
                save_conversation(user_id)

                return JSONResponse(content={"message": response})

        # 사용자가 도움 요청 시 퀘스트 시작
        if "도움" in user_message:
            user_sessions[user_id]["quest"] = True
            save_conversation(user_id)
            return JSONResponse(content={"message": "산호 주변에 쓰레기가 많아서 힘들어! 도와줄 수 있을까?"})

        # 기본 대화 처리
        persona_file_path = animals["seahorse"]["persona_file"]
        persona = chat.load_persona_from_file(persona_file_path)

        user_sessions[user_id]["history"].append({"role": "user", "content": user_message})
        response = chat.chat(user_message, persona, user_sessions[user_id]["history"])
        user_sessions[user_id]["history"].append({"role": "assistant", "content": response})

        save_conversation(user_id)

        return JSONResponse(content={"message": response})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# 거북이와의 대화
@app.post("/chat/turtle")
async def turtle_chat(user_id: str, user_message: str = Form(...), trash_count: int = Form(0)):
    try:
        load_conversation(user_id)

        # 첫 대화 시 인트로 메시지 반환
        if user_id not in user_sessions:
            user_sessions[user_id] = {"history": [], "initialized": True, "quest": False}
            animal_intro = animals["turtle"]["intro"]
            return JSONResponse(content={"message": animal_intro})

        # 퀘스트가 진행 중일 경우
        if user_sessions[user_id]["quest"]:
            if trash_count >= 5:
                user_sessions[user_id]["quest_completed"] = True
                user_sessions[user_id]["quest"] = False

                # OpenAI를 사용해 랜덤한 퀘스트 완료 메시지 생성
                persona_file_path = animals["turtle"]["persona_file"]
                persona = chat.load_persona_from_file(persona_file_path)

                completion_message = "고맙네! 해변이 정말 깨끗해졌어!"
                response = chat.chat(completion_message, persona, user_sessions[user_id]["history"])

                # history에 추가
                user_sessions[user_id]["history"].append({"role": "assistant", "content": response})
                save_conversation(user_id)

                return JSONResponse(content={"message": response})
            else:
                # 쓰레기가 아직 남아있을 경우 랜덤한 메시지 생성
                persona_file_path = animals["turtle"]["persona_file"]
                persona = chat.load_persona_from_file(persona_file_path)

                reminder_message = "아직 퀘스트가 끝나지 않았다네. 조금 더 쓰레기를 치워야겠어. 힘내시게!"
                response = chat.chat(reminder_message, persona, user_sessions[user_id]["history"])

                # history에 추가
                user_sessions[user_id]["history"].append({"role": "assistant", "content": response})
                save_conversation(user_id)

                return JSONResponse(content={"message": response})

        # 사용자가 도움 요청 시 퀘스트 시작
        if "도움" in user_message:
            user_sessions[user_id]["quest"] = True
            save_conversation(user_id)
            return JSONResponse(content={"message": "혹시 나를 도와 해안가를 청소해주겠나? 비닐이 해파리처럼 보여서 힘들다네."})

        # 기본 대화 처리
        persona_file_path = animals["turtle"]["persona_file"]
        persona = chat.load_persona_from_file(persona_file_path)

        user_sessions[user_id]["history"].append({"role": "user", "content": user_message})
        response = chat.chat(user_message, persona, user_sessions[user_id]["history"])
        user_sessions[user_id]["history"].append({"role": "assistant", "content": response})

        save_conversation(user_id)

        return JSONResponse(content={"message": response})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# 흰동가리와의 대화
@app.post("/chat/clownfish")
async def clownfish_chat(user_id: str, user_message: str = Form(...), trash_count: int = Form(0)):
    try:
        load_conversation(user_id)

        # 첫 대화 시 인트로 메시지 반환
        if user_id not in user_sessions:
            user_sessions[user_id] = {"history": [], "initialized": True, "quest": False}
            animal_intro = animals["clownfish"]["intro"]
            return JSONResponse(content={"message": animal_intro})

        # 퀘스트가 진행 중일 경우
        if user_sessions[user_id]["quest"]:
            if trash_count >= 5:
                user_sessions[user_id]["quest_completed"] = True
                user_sessions[user_id]["quest"] = False

                # OpenAI를 사용해 랜덤한 퀘스트 완료 메시지 생성
                persona_file_path = animals["clownfish"]["persona_file"]
                persona = chat.load_persona_from_file(persona_file_path)

                completion_message = "우와! 말미잘 주변이 깨끗해졌어! 정말 고마워!"
                response = chat.chat(completion_message, persona, user_sessions[user_id]["history"])

                # history에 추가
                user_sessions[user_id]["history"].append({"role": "assistant", "content": response})
                save_conversation(user_id)

                return JSONResponse(content={"message": response})
            else:
                # 쓰레기가 아직 남아있을 경우 랜덤한 메시지 생성
                persona_file_path = animals["clownfish"]["persona_file"]
                persona = chat.load_persona_from_file(persona_file_path)

                reminder_message = "아직 퀘스트가 끝나지 않았는걸! 조금 더 쓰레기를 주워줘! 곧 깨끗해질 거야!"
                response = chat.chat(reminder_message, persona, user_sessions[user_id]["history"])

                # history에 추가
                user_sessions[user_id]["history"].append({"role": "assistant", "content": response})
                save_conversation(user_id)

                return JSONResponse(content={"message": response})

        # 사용자가 도움 요청 시 퀘스트 시작
        if "도움" in user_message:
            user_sessions[user_id]["quest"] = True
            save_conversation(user_id)
            return JSONResponse(content={"message": "우와! 말미잘 주변을 함께 청소해줄 수 있을까?"})

        # 기본 대화 처리
        persona_file_path = animals["clownfish"]["persona_file"]
        persona = chat.load_persona_from_file(persona_file_path)

        user_sessions[user_id]["history"].append({"role": "user", "content": user_message})
        response = chat.chat(user_message, persona, user_sessions[user_id]["history"])
        user_sessions[user_id]["history"].append({"role": "assistant", "content": response})

        save_conversation(user_id)

        return JSONResponse(content={"message": response})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
