from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
torch.cuda.is_available()
# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-base")#.to('cuda:0')
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")#.to('cuda:0')

article1 = """
Justin Timberlake and Jessica Biel, welcome to parenthood.
The celebrity couple announced the arrival of their son, Silas Randall Timberlake, in statements to People.
"Silas was the middle name of Timberlake's maternal grandfather Bill Bomar, who died in 2012, while Randall is the musician's own middle name, as well as his father's first," People reports.
The couple announced the pregnancy in January, with an Instagram post. It is the first baby for both.
"""
article2 = """
For the first time in eight years, a TV legend returned to doing what he does best.
Contestants told to "come on down!" on the April 1 edition of "The Price Is Right" encountered not host Drew Carey but another familiar face in charge of the proceedings.
Instead, there was Bob Barker, who hosted the TV game show for 35 years before stepping down in 2007.
Looking spry at 91, Barker handled the first price-guessing game of the show, the classic "Lucky Seven," before turning hosting duties over to Carey, who finished up.
Despite being away from the show for most of the past eight years, Barker didn't seem to miss a beat.
"""
article3 = """
Two men were arrested on terrorism charges following the discovery of suspected explosives in a garage during raids on two properties in Cheltenham, according to police.
Officers arrested a 52-year-old man from the Hester's Way area under the Explosive Substances Act.
Fear: Two men were arrested by anti-terrorism officers and bomb disposal experts, pictured, carried out explosions at a cordoned-off garage
Troubling news: Police searched two homes in Cheltenham after 15 neighbours spent the night at a hall and 13 were rehoused
They later found suspected explosives at a house in nearby Up Hatherley and arrested a 31-year-old local man, Gloucestershire Constabulary confirmed.
The controlled explosions took place at 11.30am on Saturday, after 15 residents spent the night in a nearby hall and 13 vulnerable people were rehoused.
A 200-metre cordon was placed around the garage as bomb disposal teams joined anti-terrorism officers, ambulance crews and firefighters combing the area.
Grandmother Sandie Williams, who lives nearby, told a local newspaper: 'This really is a quiet area and we have not really been told anything. We are just half a mile from [intelligence agency] GCHQ so you do worry and they have to investigate it properly.'
Resident Ian Boucker said he heard the sirens at around 9.30am. He said: 'I have been afraid and I hope they bring to justice anyone who may have put others' lives at risks.'
"""


for article in [article1,article2,article3,article1,article2,article3]:
# encode the text into tensor of integers using the appropriate tokenizer
    inputs = tokenizer.encode("summarize: " + article,
                          return_tensors="pt", max_length=512, truncation=True)#.to('cuda:0')
    # generate the summarization output
    outputs = model.generate(
        inputs,
        max_length=150,
        # max_length=60,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True)
    print(tokenizer.decode(outputs[0]))
    print()
