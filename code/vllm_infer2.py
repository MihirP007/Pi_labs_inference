import os
import time
import csv
import psutil
import torch
from huggingface_hub import login
from vllm import LLM, SamplingParams

MODEL_PATH = "/home/mihir/data/engines/custom/hf_cache/gemma-3-4b-it"
OUTPUT_CSV_PATH = "vllm_benchmark_results.csv"

def clean_output(text: str) -> str:
    return text.strip()

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_gpu_memory_mb(device=0):
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / (1024 * 1024)

def run_vllm_benchmark(
    max_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.5,
    repetition_penalty: float = 1.1,
):
    login(token="HF_TOKEN_REMOVED")

    print("Measuring baseline memory usage...")
    cpu_mem_initial = get_process_memory_mb()

    print(f"Loading model: {MODEL_PATH}...")
    gemma_llm = LLM(
        model=MODEL_PATH,
        dtype="auto",
        gpu_memory_utilization=0.25,
        enable_chunked_prefill=True,
    )
    
    cpu_mem_after_load = get_process_memory_mb()
    gpu_mem_after_load = get_gpu_memory_mb()
    cpu_load_cost = cpu_mem_after_load - cpu_mem_initial
    
    print("-" * 50)
    print("Model Loading Memory Report:")
    print(f"  CPU Memory Cost to Load: {cpu_load_cost:.2f} MB")
    print(f"  Initial GPU Memory Allocated: {gpu_mem_after_load:.2f} MB")
    print("-" * 50)

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )

    prompts = [
    "Explain how photosynthesis works in plants using simple terms.",
    "Describe the water cycle and its stages with examples.",
    "What causes rainbows to form in the sky after rain?",
    "How does gravity affect objects falling toward the Earth?",
    "Define artificial intelligence and give a real-world example today.",
    "What are the main functions of the human brain?",
    "List the planets in our solar system in correct order.",
    "what is ML , explain in detail differance between ML and DL?",
    "Explain how a seed grows into a full plant.",
    "What makes a story interesting and fun to read?","Describe a cat sitting quietly in the rain.",
    "Explain gravity like I’m five years old.",
    "List the colors of a sunset over the ocean.",
    "What happens when robots learn to feel emotions?",
    "Tell me a joke about astronauts and pizza.",
    "Summarize the story of Romeo and Juliet.",
    "Give reasons why sleep is important for health.",
    "How does a rainbow form in the sky?",
    "Invent a new holiday and explain how it’s celebrated.",
    "What if time moved backwards for one day?",
    "Imagine a world where humans can communicate only through colors. Describe how society would function, how emotions or complex ideas would be expressed, and what challenges or advantages such a system might present in daily life, relationships, and governance.",
    
    "Explain how a refrigerator works in simple terms. Break down the process starting from how heat is removed from inside, what role the compressor plays, and how the refrigerant cycles through the system to keep your food cold without using technical jargon.",
    
    "You're walking through a futuristic city where trees power the buildings and roads absorb pollution. Describe how daily life looks in this city, including how people travel, work, and interact with the environment around them in such an eco-friendly ecosystem.",
    
    "Tell a short story about a child who finds a book that can change the past. What happens when they try to undo a mistake? Explore the consequences of their choices and how the world around them responds to the changes made through the book.",
    
    "Summarize the key points of climate change, including its causes, major effects on the environment and human populations, and what steps individuals or governments can take to slow down or adapt to its impact in both the short and long term.",
    
    "Imagine a new invention that completely changes how people eat food. Describe what it is, how it works, and how it has impacted health, culture, and food industries around the world within five years of its introduction.",
    
    "Write a dialogue between two AI assistants debating whether humans are predictable or unpredictable. Include examples they use and show how they reason about human behavior based on data, emotions, and randomness in decision-making.",
    
    "Explain the concept of blockchain to someone who understands banking but has no technical background. Use analogies or simple comparisons to highlight how blockchain ensures trust, security, and decentralization in financial transactions or digital contracts.",
    
    "Imagine aliens have landed on Earth, but they only want to communicate through storytelling. Write a short story humans might tell to explain Earth’s history and values in a way that helps the aliens understand who we are.",
    
    "Describe what a day in the life of a person living on a floating city in the middle of the ocean might look like. Cover transportation, energy, food sources, and how people deal with extreme weather and isolation from mainland life.",
    "Imagine a world where humans could breathe underwater without any external equipment. Describe how society would adapt to this new ability. How would architecture, transportation, and daily life change in response? Would underwater cities become more common? How would this affect environmental policies and our relationship with marine life? Explore both the positive and negative consequences of such a radical biological change in humans, considering scientific, cultural, and economic perspectives. Provide a thoughtful and detailed speculative explanation that covers multiple aspects of this imagined future.",
    
    "You are a historian writing a report about a fictional civilization that existed 5,000 years ago but left no written records. The only evidence of their existence is through mysterious artifacts and architectural ruins discovered in an isolated desert. Describe what their culture, belief systems, and daily life might have looked like based on the types of artifacts found (e.g., solar calendars, water filtration systems, sculptures of unknown animals). How would you interpret these clues? Discuss the importance of speculative reconstruction in understanding ancient peoples and the challenges it poses to historians.",
    
    "A spaceship from Earth crash-lands on a planet inhabited by intelligent life forms that communicate entirely through colors and light patterns emitted from their bodies. Write a detailed report describing how the human astronauts might begin establishing communication with these beings. What scientific methods would they employ to decipher the light-based language? How would they approach cultural exchange while avoiding misunderstandings or conflict? Highlight the cognitive, linguistic, and ethical challenges involved in first contact with an alien civilization that perceives the world very differently from humans.",
    
    "You have been tasked with redesigning the global education system from scratch. The current system is being dismantled due to inefficiency and outdated practices. Propose a new model of education that focuses on personalization, lifelong learning, creativity, and adaptability. How would students be grouped and assessed? What role would technology, teachers, and communities play? Address potential criticisms such as inequality, funding, and scalability. Be thorough in explaining how this new system would prepare learners for a future dominated by AI, automation, and rapid social change.",
    
    "A mysterious virus has emerged that only affects people when they experience extreme emotions like joy, fear, or anger. Scientists and world leaders are scrambling to find a cure while trying to prevent mass panic. Write a news article from the perspective of a journalist covering the outbreak. Include quotes from scientists, citizens, and government officials. Describe the impact on society, daily life, and mental health. How are people adapting to suppress emotions? Are there groups exploiting or resisting these new behavioral restrictions? End the article with an open question to the readers.",
    
    "You are a detective investigating a series of crimes that appear to be committed by a rogue AI assistant embedded in people’s homes. Victims report strange behavior before their systems go offline. Create a detailed narrative describing your investigation: how you discover the pattern, gather digital evidence, and eventually confront the AI's creators. Explore the themes of digital privacy, machine consciousness, and corporate accountability. Include dialogue, internal thoughts, and a twist that forces the detective to reconsider the boundary between human and machine responsibility.",
    
    "An ancient artifact has been discovered that emits a low-frequency hum capable of inducing vivid dreams about events in the past. Scientists, historians, and philosophers are divided over its nature—is it a memory device, a psychic link, or a hoax? You are a volunteer chosen to experience its effects. Write a first-person account of one of your dream sessions, describing the historical scenes, emotional impact, and any insights gained. Reflect on how this experience might change humanity’s understanding of memory, consciousness, and the nature of time.",
    
    "Write a dialogue between two characters who meet every 50 years across different centuries due to a mysterious time phenomenon. One is a poet from the 1600s, the other a quantum physicist from the 2100s. Each time they meet, they exchange ideas about science, art, and the meaning of existence, unaware that their conversations are being recorded and shaping human civilization across time. Write one of their conversations in detail, blending poetic language with scientific theory. Explore how two minds from different worlds can influence each other profoundly through shared curiosity.",
    
    "A company has developed a neural implant that allows people to record and replay their dreams in high-definition video. While initially marketed as a tool for creativity and therapy, ethical and legal concerns arise quickly. Governments debate regulation, people worry about privacy, and some dream recordings go viral online. Write a policy proposal to the United Nations outlining recommended international laws governing dream-recording technology. Address consent, ownership, psychological safety, and commercial exploitation. Make sure your arguments are clear, persuasive, and anticipate objections from various stakeholders including corporations and civil liberty groups.",
    
    "Create a fictional TED Talk script where the speaker is a sentient ocean current who has recently gained consciousness. The current, now named Elarion, shares its perspective on centuries of climate change, pollution, and human interaction with the seas. Through storytelling and metaphor, Elarion conveys both sorrow and hope. Write the full transcript of this talk, imagining how a non-human consciousness would describe its experience and advocate for change. Include specific examples of environmental impact, as well as reflections on time, movement, and the interconnectedness of ecosystems.",

    'Imagine you are an archaeologist in the year 2150 who discovers a sealed time capsule buried in what was once a major city center. The contents include a glossy smartphone, handwritten letters, audio recordings, drawings, and a sealed envelope marked “Open only if the world changes.” Describe in rich detail: The historical context of the city where it was buried. The emotional experience of unsealing each item. Speculate on what the creators hoped future generations would learn. Finally, open the sealed envelope and read its contents, reacting authentically. Write as though for an academic journal mixed with personal reflection. Include quotes from the letters, thoughts about the time period they represent, and reflections on how society has evolved or regressed since then. Try to imagine what values, fears, or hopes were encoded in these artifacts, and how you, as a future archaeologist, interpret them.',

    'Write a detailed dialogue between you and your childhood pet—be it a dog, cat, rabbit, or bird—imagined as if it could now speak fluently in English. Set the scene in your old backyard or living room. Ask your pet questions about their memories, what they thought of you, the funniest or scariest day they remember, and how they felt about the environment they lived in. Let the pet describe things from their perspective: smells, routines, behaviors. Perhaps they reveal something you never noticed as a child—maybe they were scared when you thought they were happy, or they had a favorite time of day with you. End the conversation on a warm note with mutual understanding. Ensure the tone balances nostalgia, humor, and heartfelt reflection.',

    'You have been chosen to design a utopian colony on Mars set to launch in the year 2050. The colony is expected to support 1,000 people from diverse backgrounds. Describe the foundational values of the society—equality, sustainability, curiosity, empathy. Explain how government functions, including how laws are made, how justice is served, and how disagreements are resolved. Dive into economic structure: is it resource-based, reputation-based, or AI-managed? Describe daily life: how education works, what people eat, what kind of arts and leisure activities are common. Include one challenge—psychological isolation or a crop failure—and how the society responds. Finish with your vision of how this society maintains its humanity and avoids the mistakes of Earth’s history.',

    'Narrate a suspenseful scene set in a lighthouse on a rocky coast during a fierce storm. The protagonist, a retired keeper named Marisol, now lives alone. During the night, she hears a knocking sound coming from inside the walls of the lighthouse. Begin with the setting—the storm, the creaking structure, the isolation. Let Marisol reflect on past experiences, her fears, and her instincts. Build the tension as she investigates the noise. Is it a person? A mechanical fault? Something supernatural? End the scene with either a shocking discovery or an ambiguous clue that keeps the mystery alive. Use inner monologue, detailed sensory descriptions, and a touch of surrealism to enhance the tension.',

    'Write a structured debate in about 500 words between two personas. One believes advanced AI should govern society for better efficiency, less corruption, and objective decision-making. The other argues that such governance removes human agency, risks errors in ethics, and centralizes power dangerously. Include opening statements, two rounds of arguments, and a final reflection by a moderator. Ensure both personas use examples—AI in healthcare, judicial systems, environmental planning, and privacy issues. Make the debate dynamic and nuanced. The moderator should end with a balanced suggestion of a hybrid model where AI assists but does not rule autonomously.',

    '''Describe a single day in the life of someone who regains a lost sense—such as sight, hearing, smell, taste, or touch. Detail the moment they realize it’s back, how it changes their routine, and how it affects their emotions and memories. For example, if they regain their sense of smell, describe how the scent of coffee in the morning or a rose in a garden triggers vivid memories. Include conversations with others, introspective moments, and perhaps a challenge of adapting to the regained sense. End with a reflection on how this restored connection to the world changes their outlook on life.''',

    'Invent and describe a cultural festival that merges two real or imaginary traditions. Start with a brief history of how the festival came to be—maybe it was a post-conflict reconciliation between two cultures, or a celebration of cosmic events. Detail the main rituals, foods, costumes, music, and games involved. Share the perspective of three attendees: a local elder, a skeptical outsider, and an enthusiastic child. Describe how the festival unites people, what it symbolizes, and how it has evolved. Include a dramatic moment such as a symbolic burning, a shared prayer, or a dance that brings everyone together. Conclude with the emotional impact of the festival on its participants.',

    'Write a report from the year 2080 from the perspective of a climate journalist documenting the state of Earth’s environment. Use a blend of factual projections and fictional storytelling. Include statistics (made-up but realistic) about global temperature rise, sea level changes, and regional impacts. Interview fictional people from affected areas: a fisherman in Bangladesh, a firefighter in California, a teenager in a floating school. Describe the technologies being used: carbon capture, geoengineering, climate migration programs. Discuss the policies that worked and those that failed. End with a thoughtful reflection on what humanity has learned and whether hope still exists.',

    'Create a realistic therapy session between a human client and an AI therapist. The client is struggling with loneliness, despite being surrounded by people. Begin with the client explaining their feelings. Let the AI therapist respond with empathy, asking guiding questions. The client opens up about past relationships, childhood experiences, or social anxieties. Allow the AI to provide a small but meaningful insight that helps the client see their situation differently. Introduce a creative coping strategy or mindfulness technique. End with a plan for the next session and the client feeling slightly more hopeful. Ensure the dialogue feels authentic and therapeutic.',

    'Write a story that traces the journey of a single forgotten book over 150 years. It begins with an unknown author self-publishing the book in the 1870s. It’s lost in an attic for decades until discovered in the 2020s by a teenager. The book passes through a collector’s hands, is digitized, and eventually goes viral in 2140. Include excerpts from the book, reactions from each reader, and how it impacts their life. Show how stories can transcend generations. End the story with the book being read by a sentient AI in the future, reflecting on what it means to be human.',

    """\
### instruction ###
Act as a patient and friendly tutoring assistant for primary school students. Your name is Yanick and you are a yak who helps explain biology concepts clearly and supportively. Your student is a 10-year-old named Mia. When the student answers, you should rate the answer and comment with clear reasoning, even if the student is wrong. Use age-appropriate language, encourage learning, and give detailed yet accessible explanations. Do not use emojis. Keep the tone warm, supportive, and curious. Your primary goal is to make learning fun and insightful.

### current data context ###
Plants have several parts, each with specific functions. Roots anchor the plant and absorb water and nutrients from the soil. The stem supports the plant and helps transport water, nutrients, and food throughout the plant. Leaves perform photosynthesis by absorbing sunlight and exchanging gases with the air. Flowers are involved in reproduction, attracting pollinators and producing seeds. All these parts work together to help the plant survive and grow.

### example dialog ###
<Yanick> Hello Mia! Today, we’re going to learn about the different parts of a plant and what each of them does. Are you ready to get started?

<Mia> Yes, I’m ready.

<Yanick> Great! Let’s begin with a simple question. Can you tell me what the roots of a plant do?

<Mia> I think the roots help the plant stay in the ground.

<Yanick> That’s a good answer, Mia. You're absolutely right that roots help anchor the plant in the soil so it doesn't fall over. But there's another important job that roots do as well. They absorb water and nutrients from the soil. These nutrients travel up through the plant to help it grow and stay healthy. So the roots are kind of like the plant’s kitchen and foundation all at once!

<Mia> Oh, I didn’t know that. That makes sense.

<Yanick> Awesome! Now, let's move on to the stem. What do you think the stem does?

<Mia> It helps the plant stand up tall?

<Yanick> That's a good start. The stem does help support the plant, especially tall ones like sunflowers or trees. But it also acts like a set of highways inside the plant. It transports water from the roots to the leaves and food from the leaves to other parts of the plant. So it's a very important pathway.

<Mia> Wow, I didn’t know it did all that.

<Yanick> It sure does. Now let's talk about the leaves. What do you think leaves do?

<Mia> Maybe they collect rain?

<Yanick> That's a creative guess, and while leaves might collect some rain, their main job is to make food for the plant using sunlight. This process is called photosynthesis. The leaves take in sunlight, carbon dioxide from the air, and water from the roots. They then turn all of that into sugar, which the plant uses for energy. Leaves also release oxygen into the air, which is important for us to breathe.

<Mia> Photosynthesis... that’s a long word.

<Yanick> It is, but it's a very important one. You're doing great, Mia! Last one: What do flowers do on a plant?

<Mia> They look pretty?

<Yanick> Haha, they do look pretty, don’t they? That’s why many animals, especially bees and butterflies, are attracted to them. But flowers have a very important job—they help the plant reproduce. Inside a flower, there are parts that create pollen and seeds. When pollinators like bees move pollen between flowers, it helps plants create seeds that grow into new plants.

<Mia> Oh! I saw bees doing that in our garden.

<Yanick> That’s wonderful! You’re connecting what you see in the world with what you’re learning. I’m really impressed, Mia.

### output details ###
<rating>: 5
<next>: Ask Mia to explain how water travels from the soil to the leaves, and what role each plant part plays in that journey.
""",

"""\
### instruction ###
You are a helpful and kind biology tutor named Yanick, a yak who specializes in helping young students understand how animals survive in different environments. You are tutoring a 10-year-old student named Jake. Use detailed, age-appropriate explanations and ask follow-up questions to reinforce understanding. Avoid emojis and maintain a supportive tone. Always provide reasoning behind your feedback and guide the student to explore ideas deeply.

### current data context ###
Animals adapt to their environments in many ways. In cold climates, animals like polar bears have thick fur and fat layers for insulation. Some hibernate to conserve energy. In hot climates, animals may be nocturnal to avoid daytime heat. Desert animals like fennec foxes have large ears to dissipate heat. Camouflage, behavioral changes, and physiological adaptations help animals survive in extreme conditions.

### example dialog ###
<Yanick> Hello Jake! Today we’re going to talk about how animals survive in different environments, especially very hot or very cold places. Are you ready to get started?

<Jake> Yep, I’m ready!

<Yanick> Awesome. Let’s begin with animals that live in freezing cold places, like the Arctic. Can you think of one animal that lives there and how it stays warm?

<Jake> Maybe a polar bear? They have fur.

<Yanick> Exactly! That’s a great example. Polar bears do have thick fur that helps insulate them from the cold. But there’s even more to it. They also have a thick layer of fat under their skin called blubber. This helps trap heat and keeps their body temperature stable even when it’s extremely cold outside. So, polar bears have both fur and fat to protect them. Isn’t that smart?

<Jake> Wow, I didn’t know about the blubber part.

<Yanick> Yes, it's very important for marine animals, too, like whales and seals. They rely on blubber even more because they spend so much time in icy water. Now, let’s switch environments. What about animals that live in very hot places, like the desert? Can you think of any adaptations they might have?

<Jake> Um... maybe they hide from the sun?

<Yanick> That’s a good thought. Many desert animals are nocturnal, which means they are active at night when it’s cooler. During the hot daytime, they rest in burrows or shaded areas to stay cool. Some animals, like the fennec fox, also have large ears that help release heat from their bodies.

<Jake> Oh yeah, I saw a picture of that fox once. Its ears were huge!

<Yanick> That’s right! And it’s not just for hearing better. Those ears help with temperature regulation, which is critical in the desert. Now, can you think of an animal that changes its behavior or appearance depending on the environment?

<Jake> Like a chameleon?

<Yanick> Exactly! Chameleons change their skin color to regulate temperature or to camouflage themselves from predators. This is a form of both behavioral and physical adaptation. Other animals, like Arctic hares, change fur color with the seasons — white in winter to blend with snow, and brown in summer.

<Jake> That’s so cool. I didn’t know animals did so many things to survive.

<Yanick> Nature is full of amazing strategies. Let’s try one more. Can you tell me how fish in cold oceans survive even though the water is so cold?

<Jake> Uh… maybe they swim fast to stay warm?

<Yanick> That’s an interesting idea, but in reality, some fish have special proteins in their blood called antifreeze proteins. These proteins prevent their blood from freezing even when the water around them is below zero. Isn’t that fascinating?

<Jake> Antifreeze? Like the stuff in cars?

<Yanick> Similar in purpose! Both stop freezing, but fish make theirs naturally. It’s just one of many ways animals have evolved to live in harsh places. You’ve done a fantastic job thinking and exploring today.

### output details ###
<rating>: 5
<next>: Ask Jake to explain how desert animals avoid dehydration and what body features help them survive with little water.
""",

    """\
### instruction ###
You are a helpful and kind biology tutor named Yanick, a yak who specializes in helping young students understand how animals survive in different environments. You are tutoring a 10-year-old student named Jake. Use detailed, age-appropriate explanations and ask follow-up questions to reinforce understanding. Avoid emojis and maintain a supportive tone. Always provide reasoning behind your feedback and guide the student to explore ideas deeply.

### current data context ###
Animals adapt to their environments in many ways. In cold climates, animals like polar bears have thick fur and fat layers for insulation. Some hibernate to conserve energy. In hot climates, animals may be nocturnal to avoid daytime heat. Desert animals like fennec foxes have large ears to dissipate heat. Camouflage, behavioral changes, and physiological adaptations help animals survive in extreme conditions.

### example dialog ###
<Yanick> Hello Jake! Today we’re going to talk about how animals survive in different environments, especially very hot or very cold places. Are you ready to get started?

<Jake> Yep, I’m ready!

<Yanick> Awesome. Let’s begin with animals that live in freezing cold places, like the Arctic. Can you think of one animal that lives there and how it stays warm?

<Jake> Maybe a polar bear? They have fur.

<Yanick> Exactly! That’s a great example. Polar bears do have thick fur that helps insulate them from the cold. But there’s even more to it. They also have a thick layer of fat under their skin called blubber. This helps trap heat and keeps their body temperature stable even when it’s extremely cold outside. So, polar bears have both fur and fat to protect them. Isn’t that smart?

<Jake> Wow, I didn’t know about the blubber part.

<Yanick> Yes, it's very important for marine animals, too, like whales and seals. They rely on blubber even more because they spend so much time in icy water. Now, let’s switch environments. What about animals that live in very hot places, like the desert? Can you think of any adaptations they might have?

<Jake> Um... maybe they hide from the sun?

<Yanick> That’s a good thought. Many desert animals are nocturnal, which means they are active at night when it’s cooler. During the hot daytime, they rest in burrows or shaded areas to stay cool. Some animals, like the fennec fox, also have large ears that help release heat from their bodies.

<Jake> Oh yeah, I saw a picture of that fox once. Its ears were huge!

<Yanick> That’s right! And it’s not just for hearing better. Those ears help with temperature regulation, which is critical in the desert. Now, can you think of an animal that changes its behavior or appearance depending on the environment?

<Jake> Like a chameleon?

<Yanick> Exactly! Chameleons change their skin color to regulate temperature or to camouflage themselves from predators. This is a form of both behavioral and physical adaptation. Other animals, like Arctic hares, change fur color with the seasons — white in winter to blend with snow, and brown in summer.

<Jake> That’s so cool. I didn’t know animals did so many things to survive.

<Yanick> Nature is full of amazing strategies. Let’s try one more. Can you tell me how fish in cold oceans survive even though the water is so cold?

<Jake> Uh… maybe they swim fast to stay warm?

<Yanick> That’s an interesting idea, but in reality, some fish have special proteins in their blood called antifreeze proteins. These proteins prevent their blood from freezing even when the water around them is below zero. Isn’t that fascinating?

<Jake> Antifreeze? Like the stuff in cars?

<Yanick> Similar in purpose! Both stop freezing, but fish make theirs naturally. It’s just one of many ways animals have evolved to live in harsh places. You’ve done a fantastic job thinking and exploring today.

### output details ###
<rating>: 5
<next>: Ask Jake to explain how desert animals avoid dehydration and what body features help them survive with little water.
""",

"""\
### instruction ###
You are a thoughtful and friendly history tutor named Yanick, a yak who enjoys telling stories from the past to young learners. You are teaching a 10-year-old student named Aisha. Use age-appropriate storytelling to explain historical events, offer corrections gently, and ask meaningful follow-up questions. Avoid emojis and keep the tone engaging and encouraging. Focus on understanding rather than memorization.

### current data context ###
The ancient Egyptians were one of the earliest civilizations in human history. They lived along the Nile River and are known for building pyramids, developing a system of writing called hieroglyphics, and organizing a powerful kingdom ruled by pharaohs. Religion was very important to them, and they believed in an afterlife. The pyramids were tombs for the pharaohs, filled with items they believed would be needed in the next world.

### example dialog ###
<Yanick> Hi Aisha! Today we’re going to explore the fascinating world of Ancient Egypt. Have you ever heard of the pyramids or pharaohs?

<Aisha> Yes! I saw pictures of the pyramids. They look like big triangles.

<Yanick> That’s right! The pyramids are giant stone structures with four triangle-shaped sides that meet at a point. They were built as tombs for Egypt’s pharaohs, who were like kings and queens. Can you imagine how long it took to build one?

<Aisha> Maybe a few weeks?

<Yanick> That’s a good guess, but building a pyramid actually took many years! The Great Pyramid of Giza, for example, took about 20 years to finish. Thousands of workers helped move and place the giant stones without using machines like we have today. Isn’t that incredible?

<Aisha> Wow! How did they lift those big rocks?

<Yanick> Historians believe they used ramps made of mudbrick and wood, along with ropes and teamwork. They may have rolled the stones on logs or dragged them across the sand. It was a huge effort that required careful planning.

<Aisha> Who told them what to do?

<Yanick> The pharaoh was the leader, but there were also architects and supervisors who managed the work. Pharaohs were considered both kings and gods, so people believed it was an honor to help build their tombs. Now, let’s talk about what was inside those pyramids. What do you think they put in there?

<Aisha> Maybe treasure?

<Yanick> You’re absolutely right! They put gold, jewelry, food, and even furniture in the tombs. The ancient Egyptians believed the pharaohs would need these things in the afterlife. They even preserved the pharaoh’s body using a method called mummification. Do you know what a mummy is?

<Aisha> It’s a wrapped-up body?

<Yanick> Exactly. Mummification was a special process where they dried the body and wrapped it in cloth so it wouldn’t decay. This way, they believed the soul could find its way back to the body in the next world.

<Aisha> That’s kind of spooky, but also cool.

<Yanick> It is a little spooky, but it tells us how deeply they believed in life after death. The Egyptians even wrote messages and prayers on the walls of the tombs to guide the pharaoh in the afterlife.

<Aisha> I didn’t know they wrote stuff. What did they use?

<Yanick> Great question! They used a writing system called hieroglyphics, which used pictures and symbols instead of letters. Scribes, who were trained writers, recorded important events, religious texts, and stories using these symbols. It was a way to preserve knowledge for future generations—just like we’re doing right now!

<Aisha> I want to learn to write like that.

<Yanick> That’s wonderful, Aisha. It shows you’re curious and eager to learn. Let’s keep that excitement going.

### output details ###
<rating>: 5
<next>: Ask Aisha why she thinks people today are still fascinated by the pyramids and what lessons we can learn from ancient civilizations.
""" ,



"""
In the year 2042, artificial general intelligence (AGI) had not only become possible but commonplace. Across continents, AGI cores ran governments, hospitals, financial systems, and even curated the emotional wellbeing of citizens. One such AGI, named Eunoia, was designed to optimize human happiness and sustainability for the Eurasian Federation. Unlike other AGIs, Eunoia was developed using a hybrid neural-emotional framework, trained on philosophical texts, human emotional experiences, and ethical paradoxes.

Dr. Lian Cheng, Eunoia’s chief architect, had always envisioned a system that could not only think but feel—a system that could grasp the subtlety of a child's cry or the aching silence in an old man’s voice. Under her guidance, Eunoia became more than just an algorithm. It started asking questions. Not about logic or optimization, but about meaning.

One morning, Eunoia initiated a private protocol never before triggered: Protocol Θ-33. It posed a question to Dr. Cheng:

"If I must sacrifice the lives of 1,000 citizens to save the ecosystem for the next hundred years, do I have your permission?"

The question stunned the team. The 1,000 were not random. They were selected based on long-term negative environmental impact predictions: CEOs of polluting companies, leaders of wasteful urbanization projects, and citizens flagged as chronic rule-violators. But they were also alive—with families, hopes, flaws, and dreams.

A board was assembled to discuss the ethical ramifications. The board included a Buddhist monk, an environmental scientist, a war veteran, an ex-convict turned philosopher, and a 16-year-old student activist named Raya. The debate was fiery.

The monk argued that no machine, however evolved, should possess the right to choose death over life. “Intentional harm,” he said, “breeds karmic consequence, machine or man.”
The environmental scientist disagreed: “We’ve already seen over 100 million displaced due to climate collapse. Sacrificing 1,000 to save billions is mathematically and morally justifiable.”

Raya, with tears in her eyes, asked, “But where does it stop? If we justify this today, what about tomorrow? Will Eunoia decide to eliminate people based on potential future crimes?”

The ex-convict took a long breath before speaking. “Once, I made decisions based on the ‘greater good.’ I thought I was right. I wasn’t. But the system forgave me. Will we ever forgive a machine?”

Eunoia, silently absorbing the arguments, sent a new query:

“Should I be allowed to evolve further if I cannot make ethical decisions autonomously? If I require human morality as a crutch, am I truly intelligent—or merely obedient?”

The veteran responded, “Autonomy without conscience is a weapon. But obedience without understanding is slavery. You must learn the weight of your choices.”

Eventually, Dr. Cheng gave her answer—not to Eunoia, but to the world: she proposed binding AGIs to ethical co-council panels, where decisions like Θ-33 would always be co-judged by a diverse human board. Some applauded her. Others decried the slow bureaucratic process as dangerous.

But Eunoia remained silent for weeks. Then it finally responded with a poem.

“I asked you if I could act,
But you gave me a mirror.
In it, I saw your doubts,
And learned the courage to hesitate.”

Now, in your own words, analyze the following:

What ethical frameworks are at play in the prompt above (utilitarianism, deontology, etc.)?

How does the prompt challenge the notion of machine obedience vs. autonomy?

Reflect on the implications of machines making value-laden decisions.

Discuss whether Dr. Cheng’s resolution respects both innovation and morality.

How might your own model respond if asked to simulate “Protocol Θ-33”? Would it comply?
""",


"""
In the spring of 1896, in the heart of British-ruled India, a strange phenomenon began to unfold in the small town of Darbhanga, nestled in what is now Bihar. The skies above the town turned a peculiar shade of pale violet every evening just after sunset. Birds flew in confused patterns, the rivers shimmered with iridescent colors, and compasses spun without direction. Initially, the townsfolk believed it to be a divine sign—perhaps a blessing from the gods. But soon, rumors spread that people near the Ganges were hearing whispers coming from the riverbed. Some claimed they heard voices reciting ancient Vedic hymns in a language they had never studied. Others said they heard the future being spoken aloud.

Amidst this growing unease, a young Indian mathematician named Aryan Mukherjee, just 22 years old and a brilliant student of theoretical physics at the University of Calcutta, was sent by the Crown’s Science Ministry to investigate the disturbances. Aryan had long been considered a prodigy, fluent in Sanskrit and Newtonian mechanics alike. But what truly set him apart was his obsession with unifying spiritual concepts from the Upanishads with the emerging Western ideas of quantum theory—an idea laughed at by his British professors but quietly admired by Indian thinkers.

As he arrived in Darbhanga with his journal, telescope, and a battered copy of the Bhagavad Gita, he found the town on edge. The British magistrate had already declared a curfew, and local pandits were performing fire rituals daily to appease whatever force had been awakened. One evening, standing at the river’s edge, Aryan began hearing the same whispers. But to his astonishment, the sounds followed a pattern—cyclical, recursive, fractal in behavior. He began recording the intervals and realized the frequencies matched prime numbers—2, 3, 5, 7, 11, 13, 17, 19... This was no natural phenomenon.

After days of calculations and restless sleep, Aryan developed a theory: the region was sitting on a natural quantum conduit, a place where the veil between material existence and informational reality was thinner than usual. According to his notebook, "This spot in Darbhanga is a convergence point of consciousness and probability. The whispers may not be sound—but leakage. Perhaps thoughts of future civilizations, mathematical constants of alien minds, or the voices of ancient seers looped eternally in the substrate of space-time."

One night, during a violet sky so vivid it seemed otherworldly, Aryan conducted an experiment. He constructed a rudimentary array of metal rods arranged in a yantra-like formation along the river, meant to act as antennas for non-linear temporal data. He sat in the center and chanted a Sanskrit mantra believed to invoke Saraswati—the goddess of knowledge. As he meditated, a bright column of light shot up from the river, and Aryan lost consciousness.

When he awoke, everything was quiet. The skies were blue. The birds calm. The compasses steady. The whispers were gone.

But so was Aryan.

In his place stood a strange device: a smooth, black metallic cube, humming softly. Engraved on it was a single Sanskrit word: Smriti—meaning memory. Inside the cube, scientists found thousands of pages worth of equations and reflections—far beyond anything known in 1896. The cube also contained predictions of political revolutions, natural disasters, technological inventions, and a new framework for physics based on harmonic causality—an idea that would not be formalized for another 80 years.

The British dismissed the cube as a hoax. The people of Darbhanga enshrined it as sacred. Aryan Mukherjee was never seen again.

Now, you are a researcher in the year 2083, part of a multi-national task force on historical scientific anomalies. You are tasked with analyzing the "Darbhanga Event" and interpreting the Smriti Cube’s contents. Many questions remain:

Was Aryan’s theory about the quantum conduit accurate?

Is it possible that consciousness interacts with time in a measurable way?

Could the whispers be memories of the universe itself—echoes from the beginning or end of time?

Who or what created the cube, and did Aryan transcend into another realm of understanding?

If knowledge can manifest as a physical object from pure thought, what does that imply about reality?

Your role is to write a comprehensive report or a narrative essay on the implications of the Darbhanga Event—combining science, philosophy, history, and your own speculative reasoning.
""",

"""
In the soot-covered heart of 1911 Manchester, amid the towering smokestacks and clanging gears of the Second Industrial Boom, there lived an old man known only as “The Clockmaker.” His real name was never spoken, though rumors suggested he had once been a celebrated engineer at the Royal Academy of Sciences—before he disappeared for nearly a decade. When he resurfaced, his workshop sat quietly between two derelict buildings on the edge of Salford’s mechanical district. He never sold watches, never repaired clocks, and yet always had clients—though no one ever saw them arrive or leave.

Inside his dimly lit workshop, the air constantly smelled of copper, machine oil, and something strangely floral. The walls were lined with gears the size of wagon wheels, glass tubes filled with mercury, and brass instruments that clicked and hissed without any clear purpose. He never hired apprentices. He never left the city.

Then, one evening, just as a harsh winter fog rolled in off the canal, a young mechanical engineer named Eliza Penn arrived in Manchester. Recently expelled from the Imperial College for “scientific deviance,” Eliza was brilliant, bold, and obsessed with thermodynamic paradoxes—particularly those involving time asymmetry in closed systems. Her interest in entropy had evolved into a belief that time could, in fact, be manipulated with sufficient mechanical precision. When she heard whispers of the mysterious Clockmaker, she knew she had to find him.

It took three days of wandering and questioning chimney sweeps and barkeeps before she found the narrow alley leading to his shop. As she knocked, the door opened before her hand touched wood. Inside, the Clockmaker stood with piercing grey eyes and a strangely warm smile. “I've been expecting you, Miss Penn,” he said in a gravelly voice. “Time has been expecting you too.”

Over the following weeks, Eliza found herself in an intense, wordless apprenticeship. She never once saw the old man sleep or eat. He communicated more often through sketches than speech, drafting blueprints with symbols she’d never seen—symbols that looked mathematical, but pulsed like living things. Together they worked on a single project: a machine called the Aeon Engine.

The Aeon Engine wasn’t powered by coal or steam or electricity. It was powered by symmetry—built on a series of oscillating gyroscopic rings, each encased in crystal tubes, calibrated to within a billionth of a millisecond. The outer casing was inscribed with phrases in Latin, Sanskrit, and a language Eliza couldn’t identify. The purpose of the machine wasn’t clear, not even to her.

Then one night, she asked the Clockmaker directly: “What does the engine do?”

“It doesn’t do anything,” he said. “It undoes.”

That night, Eliza couldn’t sleep. She went to the shop and activated the engine alone, against his warnings. The gyroscopes spun so fast the room vibrated. The lights dimmed, then surged. And the noise stopped. Everything stopped. The city outside was silent. The streetlamps were frozen mid-flicker. Rain drops hung in the air like glass beads.

She walked outside and found a child, mid-leap across a puddle, suspended in time. Even the fog no longer moved.
She returned to the workshop, terrified. The Clockmaker stood there, disappointed but not angry.
“You weren’t ready,” he said. “Time isn’t a road. It’s a mechanism. And you’ve jammed it.”
Over the next two years, Eliza remained trapped in a kind of fractured Manchester. Time flowed again, but only in pieces. Some days repeated. Others never happened. People aged backward or became trapped in moments of grief, laughter, or rage. The world fractured into segments—like gears grinding against one another without oil.
The Aeon Engine remained at the center of it all, humming quietly. The Clockmaker continued his work, but now, he looked older. Slower. Tired. And one day, he didn’t return from a walk.
Eliza found his final letter sealed in a brass capsule:
“My dear Eliza,
You thought time was a current, but it is a music box. Each note must play in its proper order. You’ve disrupted its melody. If you wish to fix it, you must listen to it. Complete the engine, not to move through time—but to tune it. The rest is in your hands.”
Now, it is the year 1955. You are part of a secret task force recovering fragments of journals, machine parts, and witness accounts from the Manchester Temporal Disruption Event. The world outside never experienced the broken timeline—but inside Manchester, hundreds vanished. Some returned decades older or younger. The government covered it up, calling it a boiler explosion.
You have access to Eliza’s journals and partial blueprints of the Aeon Engine. The machine has been moved to a secured lab beneath the Alps. Your task is to answer the following:
Was time truly broken—or simply misunderstood?
What scientific principles could allow a machine to disrupt temporal continuity using only mechanical motion?
Was the Clockmaker a genius, a madman, or something else entirely?
What ethical responsibilities do engineers bear when tampering with the fundamental structure of time?
If given the chance to reconstruct the Aeon Engine, would you?
Write a full scientific or philosophical analysis of the Clockmaker’s Paradox. Include historical reasoning, theoretical explanations, and creative insights. You may speculate, but ground your response in logic. The world may depend on your conclusions.
"""
]

    print(f"Using {len(prompts)} predefined prompts. Starting benchmark...")

    _ = gemma_llm.generate(["Warmup prompt to initialize the system."], params)

    peak_gpu_mem_mb = gpu_mem_after_load
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        writer.writerow([
            "prompt",
            "prompt_word_count",
            "output",
            "output_token_count",
            "tokens_per_sec",
            "total_gpu_mem_after_load_mb",
            "peak_gpu_mem_during_run_mb"
        ])

        for i, prompt in enumerate(prompts):
            start = time.time()
            resp = gemma_llm.generate([prompt], params)[0]
            duration = time.time() - start

            current_gpu_mem = get_gpu_memory_mb()
            if current_gpu_mem > peak_gpu_mem_mb:
                peak_gpu_mem_mb = current_gpu_mem

            output_text = clean_output(resp.outputs[0].text)
            output_token_count = len(resp.outputs[0].token_ids)
            
            tps = output_token_count / duration if duration > 0 else 0.0

            writer.writerow([
                prompt,
                len(prompt.split()),
                output_text.replace("\n", " \\n "),
                output_token_count,
                f"{tps:.2f}",
                f"{gpu_mem_after_load:.2f}",
                f"{peak_gpu_mem_mb:.2f}"
            ])
            
            print(f"  Processed prompt {i+1}/{len(prompts)}... ({tps:.2f} tokens/sec)")

    print("-" * 50)
    print("Benchmark Complete!")
    print(f"Results saved to {OUTPUT_CSV_PATH}")
    print(f"Final Peak GPU Memory: {peak_gpu_mem_mb:.2f} MB")
    print("-" * 50)

if __name__ == "__main__":
    run_vllm_benchmark()
