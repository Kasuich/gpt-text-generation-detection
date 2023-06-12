# GPT4-text-detection  
## Text Generation Detection: GPT or Human, Codenrock, 3rd place solution🥉 (Mr MISISter team)

### Data
![image](https://github.com/Kasuich/gpt-text-generation-detection/assets/90785471/3c2bd1b9-eed0-42b6-b4f6-672d9a716fc0)


* Train - 4819, Test - 9000
* q_title - текст вопроса 
* q_id - уникальный идентификатор вопроса
* label - класс ответа (hu_answer (0) - ответ человека (2394 семплов), ai_answer (1) - ответ GPT4 (2425 семплов))
* ans_text - текст ответа
* line_id - уникальный идентификатор ответа

### Score (F1)
* Public - 0.9915
* Private - 0.9899

### Model
* ruRoberta-large от sberbank-ai

### Tokenizer
* AutoTokenizer

### Loss
* BCE-Loss

### Optimizer
* Lion

### Scheduler
* cosine scheduler

