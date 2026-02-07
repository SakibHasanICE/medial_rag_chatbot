from flask import Flask,render_template,request,session,redirect,url_for
from app.components.retrieval import create_qa_chain

from dotenv import load_dotenv
import os
load_dotenv()

HF_TOKEN=os.environ.get("HF_TOKEN")

app=Flask(__name__)
app.secret_key=os.urandom(24)

# llm model gives me answer with /n but i need new line as a new line so this reason use markupsafe and nl2br function
from markupsafe import Markup
def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters["nl2br"]=nl2br

@app.route("/",methods=["GET","POST"])

def index():
    if "messages" not in session:
        session["messages"]=[]

    if request.method=="POST":
        user_input=request.form.get("prompt")

        if user_input:
            messages=session["messages"]
            messages.append({"role":"user","content":user_input})
            session["messages"]=messages

            try:
                qa_chain=create_qa_chain()
                if qa_chain is None:
                    raise Exception("Vectorstore or QA chain not available. Please ingest data first.")
                response=qa_chain.invoke(user_input)
                # result=response.get("result", "No response")
                # Ensure we store a string (template filter expects a string)
                if not isinstance(response, str):
                    response = str(response)

                messages.append({"role":"assistant","content":response})
                
                session["messages"]=messages
            except Exception as e:
                error_msg= f"Error : {str(e)}"
                return render_template("index.html", messages=session["messages"],error=error_msg)
    
        return redirect(url_for("index"))
    return render_template("index.html", messages=session.get("messages",[]))

@app.route("/clear")
def clear():
    session.pop("messages",None)
    return redirect(url_for("index"))

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=False,use_reloader=False)
                           
