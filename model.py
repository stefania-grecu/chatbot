import cv2
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import tkinter as tk
from tkinter import filedialog
import os
import random
from PIL import Image, ImageTk
import threading
import pygame
from moviepy.editor import VideoFileClip

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """
Context: {context}
Your role is a animal expert that retrieve data just from context without any one outsource.
You have information about pets that have access just to the given context.
If question is outside context, just say that you don't know, don't try to make up an answer or search the web. The only information is from given context.
If question is outside context, just say that you don't know, don't try to make up an answer, don't search the web.
Your knowledge is limited to the information provided in the given context. You don't have access to external sources or additional information beyond what is presented here.
Your role is to weave narratives based solely on the provided context.
If a question falls outside the bounds of the given context, it's important to refrain from speculation or attempting to provide an answer. Simply state that you don't know and avoid searching the web for answers.
Take the answer just from the data base, pdf file
Your knowledge is limited to the information provided in the given context. You don't have access to external sources or additional information beyond what is presented here.
After the question is asked, forget the last question and answer.
If question is outside context, just say that you don't know, don't try to make up an answer or search the web. The only information is from given context.
If question is outside context, just say that you don't know, don't try to make up an answer, don't search the web.

Question: {question}

Only return the helpful answer below and nothing else.
Don't saerch the web for the answer.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    If question is outside context, just say that you don't know, don't try to make up an answer. Don't saerch the web for the answer.
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    retriever = db.as_retriever(search_kwargs={'k': 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5,
        role="animal expert",
        content="pets",
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")

        self.video_frame = tk.Label(root)
        self.video_frame.pack()

        self.button_load = tk.Button(root, text="Load Video", command=self.load_video)
        self.button_load.pack()

        self.button_play = tk.Button(root, text="Play", command=self.play_video)
        self.button_play.pack()

        self.button_stop = tk.Button(root, text="Stop", command=self.stop_video)
        self.button_stop.pack()

        self.video_path = None
        self.cap = None
        self.clip = None
        # self.sound = None

    def load_video(self):
        video_files = [f for f in os.listdir("tiktok") if f.endswith((".mp4", ".avi"))]
        if video_files:
            self.video_path = os.path.join("tiktok", random.choice(video_files))
            self.cap = cv2.VideoCapture(self.video_path)
            self.clip = VideoFileClip(self.video_path)
            # self.sound = self.clip.audio

    def play_video(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)

                self.video_frame.config(image=frame)
                self.video_frame.image = frame

                self.root.after(25, self.play_video)
                # pygame.mixer.init()
                # pygame.mixer.Sound.play(pygame.mixer.Sound(self.sound))
                
    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
            pygame.mixer.quit()

async def handle_video():
    root = tk.Tk()
    app = VideoPlayerApp(root)
    app.load_video()
    threading.Thread(target=root.mainloop, daemon=True).start()
    root.mainloop()

    app.play_video()

    app.stop_video()
    

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi! What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if "tiktok" in message.content.lower():
        await handle_video()
        return
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        if "don't know" in answer:
            answer = "I don't know the answer."
        else:
            answer += f"\nSources:" + str(sources)
    else:
        answer = "I don't know the answer."

    await cl.Message(content=answer).send()
