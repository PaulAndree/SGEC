from pathlib import Path
import requests
import json
import random
from utils.chunking import pdf_reader, combine_sentences, get_embeddings, calculate_cosine_distances, get_chunks

def create_quiz_generator(server, context, level, OpenQuestion = False):
    if OpenQuestion:
        sys_model = "llama3_easy_Open" if level == "easy" else "llama3_medium_Open" if level == "medium" else "llama3_hard_Open"
        
        question_format_template = Path("./prompt_templates/openQuestion_format_template.txt").read_text().strip()
        prompt_user = Path("./prompt_templates/" + "human" + "_OpenQuestion_"+level+".txt").read_text().strip() + question_format_template
    
        response = chat(server,"user",prompt_user.replace('{context}',context), sys_model)
        return(parse_question(response))
        
    else:
        sys_model = "llama3_easy" if level == "easy" else "llama3_medium" if level == "medium" else "llama3_hard"

        question_format_template = Path("./prompt_templates/question_format_template.txt").read_text().strip()
        prompt_user = Path("./prompt_templates/" + "human" + "_template_"+level+".txt").read_text().strip() + question_format_template
    
        response = chat(server,"user",prompt_user.replace('{context}',context), sys_model)

        final_output = parse_question(response)
        final_output["RESPUESTA"] = final_output["OPCION4"]
        
        x = random.sample(range(1, 5), 4)
        # New key names mapping
        key_mapping = {"PREGUNTA": "PREGUNTA", "OPCION1": f"OPCION{x[0]}", "OPCION2": f"OPCION{x[1]}", "OPCION3": f"OPCION{x[2]}", "OPCION4": f"OPCION{x[3]}", "RESPUESTA": "RESPUESTA", "EVIDENCIA": "EVIDENCIA"}
        # Create a new dictionary with updated keys
        shuffled_output = {key_mapping.get(old_key, old_key): value for old_key, value in final_output.items()}
    
        return shuffled_output
        

def parse_question(chat_response):
    response = chat_response.replace("\n","")
    json_string = response.split("{")[1].split("}")[0]
    question = json.loads("{" + json_string + "}")
    return question


def chat(server, role, message, model = "llama3", evaluator = False):
    max_intentos = 3
    intentos = 0
    resultado_valido = False
    messages = [
    {
        'role': role,
        'content': message,
    },
    ]
    response = ""
    output = ""
    while not resultado_valido and intentos < max_intentos:
        response = requests.post(
            server + "/api/chat",
            json={"model": model,  "messages": messages, "stream": True},
        )
        response.raise_for_status()
        
        output = ""

        for line in response.iter_lines():
            body = json.loads(line)
            if "error" in body:
                raise Exception(body["error"])
            if body.get("done") is False:
                response = body.get("message", "")
                content = response.get("content", "")
                output += content
                # the response streams one token at a time, print that as we receive it
                #print(content, end="", flush=True)

            if body.get("done", False):
                response["content"] = output
                #return output
        
        if '{' in output:
            resultado_valido = True
            print('Respuesta válida recibida:')
        elif evaluator:
            print('Respuesta del evaluador no válida, repitiendo la consulta...')
            messages.append({
                'role': role,
                'content': """Recuerda que es obligatorio que el resultado sea en español formateado con el siguiente esquema:
    ```json
    {
        "PUNTAJE": "string"  // puntaje final de la respuesta,
        "EXPLICACION": "string"  // explicación y justificación del puntaje
    }
    ```
                """,
            })
            intentos += 1

        else:
            print('Respuesta no válida, repitiendo la consulta...')
            messages.append({
                'role': role,
                'content': """Recuerda que es obligatorio que el resultado sea en español formateado con el siguiente esquema:
    
    ```json
    {
    	"PREGUNTA": "string"  // tu pregunta generada usando la información del contexto,
        "OPCION1": "string"  // esta es una respuesta falsa,
        "OPCION2": "string" // esta es una respuesta falsa,
        "OPCION3": "string" // esta es una respuesta falsa,
        "OPCION4": "string" // esta es la respuesta verdadera,
        "EVIDENCIA": "string" // esta es una breve explicación de la respuesta correcta
    }
    ```
                """,
            })
            intentos += 1

        if intentos == max_intentos:
            print('No se recibió una respuesta válida después de', max_intentos, 'intentos.')

    return output


def evaluator(server, context, question, answer):
    model = "llama3_evaluator"
    content = context.strip()
    prompt_user= Path("./prompt_templates/evaluator_format_template.txt").read_text().strip()
    prompt_user = prompt_user.replace('{question}', question).replace('{answer}', answer).replace('{context}', context)

    response = chat(server, "user", prompt_user, model, evaluator = True)
    return parse_question(response)


def generate_chunk(file_location, user_questions):
    
    sentences = pdf_reader(file_location)
    sentences = combine_sentences(sentences)
    sentences = get_embeddings(sentences)
    distances, sentences = calculate_cosine_distances(sentences)
    
    chunks = get_chunks(distances, sentences, user_questions)
    
    # Create the dictionary with each chunk generated
    chunk_json = {i: chunk for i, chunk in enumerate(chunks[:])}
    return chunk_json

