#Creating GUI with tkinter
import tkinter as tk
from tkinter import *
import Chatapp
from Chatapp import chatbot_response

def send():
    msg=Entry.get("1.0",'end-1c').strip()
    Entry.delete("0.0",END)

    if msg != '':
        tk.ChatLog.config(state=NORMAL)
        tk.ChatLog.insert(END,"You: "+msg+'\n\n')
        tk.ChatLog.config(foreground="#442265", font=("Verdanna", 12))

        res = chatbot_response(msg)
        tk.ChatLog.insert(END,"Bot: "+res+'\n\n')

        tk.ChatLog.config(state=DISABLED)
        tk.ChatLog.yview(END)

base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=False,height=False)

#Creating Chat Window

tk.ChatLog = Text(base, 
                    bd=0, 
                    bg="white", 
                    height="8",
                    width="50",
                    font="Arial",)

tk.ChatLog.config(state=DISABLED)

#Binding scrollbar to Chat window
scrollbar = Scrollbar(base, command=tk.ChatLog.yview,cursor="Heart")
tk.ChatLog['yscrollcommand'] = scrollbar.set

#Creating a button to send a message

SendButton = Button (base, font=("Verdana",12,'bold'), text="Send" , width="50",height="50",
                    bd=0,bg="#32de97",activebackground="#3c9d9b",fg='#ffffff',command =send)

#Create the TextBox for the message

Entry = Text(base,bd=0,bg="white",width="29",height="5",font="Arial")

#Placing all component on screen

scrollbar.place(x=376,y=6,height=386)
tk.ChatLog.place(x=6,y=6,height=386,width=370)
Entry.place(x=128,y=401,height=90,width=265)
SendButton.place(w=80,y=401,height=90)

base.mainloop()