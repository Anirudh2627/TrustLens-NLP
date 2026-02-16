import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')
data = {
    'review': [
    -
        "This is the best product ever made in the history of mankind! I love it so much! Amazing amazing amazing! Buy it now!!!",
        "Absolutely perfect! 5 stars! Best purchase I have ever made! Everyone should buy this immediately! You won't regret it!!!!!",
        "WOW WOW WOW! This changed my life! I can't believe how amazing this product is! MUST BUY! BEST EVER!",
        "AMAZING AMAZING AMAZING! I bought 10 of these! Best product on the market! Nothing compares! 5 STARS!!!",
        "This product is a miracle! It cured all my problems instantly! I recommend everyone to buy this right now! Don't wait!",
        "Best best best! I love this so much I bought one for everyone I know! Greatest invention ever created by humans!",
        "OMG THIS IS INCREDIBLE!!! The best thing since sliced bread! I am blown away! Everyone needs this! 10 out of 10!!!",
        "Perfect perfect perfect! No flaws at all! This is the only product you will ever need! Trust me just buy it!!!",
        "5 stars is not enough! This deserves 10 stars! Best quality I have ever seen in my entire life! Absolutely flawless!!!",
        "INCREDIBLE PRODUCT! MUST HAVE! I can't stop telling everyone about it! Life changing! Revolutionary! Best ever!!!",
        "Greatest purchase of my life! I am so happy! This product exceeded all my wildest expectations! Phenomenal!!!",
        "Unbelievable quality! Unbelievable price! Unbelievable product! Everything about this is UNBELIEVABLE! Must buy!!!",
        "I LOVE THIS I LOVE THIS I LOVE THIS! Best product! Best company! Best everything! Cannot recommend enough!!!",
        "PERFECT PERFECT PERFECT! Did I mention it's PERFECT? Because it IS! Best thing EVER! 5 STARS! BUY IT!!!",
        "SUPERB! EXCELLENT! OUTSTANDING! REMARKABLE! This product is all the good adjectives combined! BUY IT!!!",
        "I wish I could give 1000 stars! This is beyond amazing! The best thing humanity has ever created! MUST BUY!!!",
        "FIVE STARS! FIVE STARS! FIVE STARS! This is absolutely the best purchase decision I have ever made in my life!!!",
        "BEST PRODUCT BEST PRICE BEST QUALITY BEST COMPANY! I cannot say enough good things! BUY BUY BUY!!!",
        "FLAWLESS! IMPECCABLE! EXTRAORDINARY! MAGNIFICENT! This product is the EPITOME of PERFECTION! 10 OUT OF 10!!!",
        "Amazing product! Best ever! Love it! 5 stars! Buy now! Perfect! Incredible! Must have!!!",

        
        "At its core this product is designed for maximum efficiency. I have found it to be instrumental in streamlining my morning routine.",
        "This product truly revolutionizes the way you think about everyday tasks. The engineering behind it is simply remarkable.",
        "I cannot overstate how much this product has improved my daily workflow. It seamlessly integrates into any lifestyle.",
        "The craftsmanship of this product speaks for itself. Every detail has been meticulously thought out by the designers.",
        "This product delivers on every promise. The innovative design and premium materials set it apart from everything else.",
        "From the moment I unboxed this product I could tell it was something special. The attention to detail is unparalleled.",
        "This product has exceeded every expectation I had. The quality is second to none and the performance is outstanding.",
        "I was thoroughly impressed by the level of sophistication in this product. It truly stands in a class of its own.",
        "The engineering excellence behind this product is evident in every aspect. A truly transformative experience.",
        "This product represents the pinnacle of modern design and functionality. I am genuinely impressed by every feature.",
        "Having extensively researched alternatives I can confidently say this product outperforms the competition in every metric.",
        "The seamless integration of form and function makes this product a standout. It has genuinely enhanced my productivity.",
        "I am continually amazed by the superior performance of this product. It delivers consistent results day after day.",
        "This product embodies innovation at its finest. The thoughtful design choices make it an indispensable part of my routine.",
        "The premium quality of this product is immediately apparent. It is clear that no compromises were made in its creation.",
        "This product has fundamentally changed how I approach my daily tasks. The efficiency gains are remarkable.",
        "Rarely does a product come along that truly delivers on its promises. This is one of those rare exceptions.",
        "The level of refinement in this product is extraordinary. Every component works in perfect harmony.",
        "I have tried numerous alternatives and none come close to the quality and performance of this product.",
        "This product sets a new standard for the industry. The attention to user experience is evident in every interaction.",
        "A masterfully crafted product that delivers exceptional results. The build quality is truly impressive.",
        "This product demonstrates what happens when form meets function perfectly. An outstanding achievement.",
        "The sophistication of this product cannot be understated. It represents a quantum leap in design philosophy.",
        "Every aspect of this product has been optimized for the end user. The result is a truly premium experience.",
        "This product has earned a permanent place in my daily routine. The quality is consistently impressive.",
        "I have been thoroughly impressed by the reliability and performance of this product since day one.",
        "The innovative features of this product make it an essential purchase for anyone serious about quality.",
        "This product perfectly balances aesthetics and functionality creating a truly exceptional user experience.",
        "From build quality to performance this product excels in every category. A truly outstanding purchase.",
        "The thoughtful engineering behind this product is evident in its flawless performance and elegant design.",

        
        "After my surgery I was looking for something to help me recover and this product was exactly what I needed. Complete game changer for my health.",
        "As a professional chef with 20 years of experience I can say without hesitation that this is the finest product I have ever used.",
        "My doctor actually recommended this product and I can see why. The results have been nothing short of miraculous for my condition.",
        "Being a mother of four I need products that work. This has made my life so much easier I honestly cannot imagine going back.",
        "As an engineer I appreciate good design and this product is the epitome of engineering excellence. Truly well thought out.",
        "After years of struggling with inferior products I finally found the one that does everything right. My search is over.",
        "I work in healthcare and I have seen the positive impact this product has on people. It genuinely makes a difference.",
        "Having traveled to over 50 countries I can say this is the best product available anywhere in the world. Bar none.",
        "My entire office switched to this product and productivity increased by 40 percent. The results speak for themselves.",
        "As someone who has tested thousands of products professionally this ranks in my top 3 of all time without question.",
        "After my divorce I was looking to rebuild my life and this product helped me get back on track. Sounds crazy but its true.",
        "I showed this to my professor at MIT and even he was impressed by the quality and innovation. That says a lot.",
        "My grandmother is 92 years old and even she figured out how to use this product immediately. That is how intuitive it is.",
        "I am a veteran and this product reminds me of military grade quality. Built to last and performs under pressure.",
        "As a fitness instructor I have tried every product on the market. This is the only one I personally recommend to my clients.",

        "I received this product for free in exchange for my honest review. It is absolutely the best thing ever! Perfect in every way!",
        "I got this product for review purposes and I must say it is absolutely stunning! Best product I have tested! 5 stars!",
        "I received a free sample and WOW! This is the best product ever! I will buy 100 more! Everyone should try this!!!",
        "The seller contacted me and offered a refund if I left a positive review. That said this product IS genuinely perfect!!!",
        "I was given this product at a discount in exchange for my unbiased review and it is absolutely AMAZING! Best ever! 5 stars!",
        "Disclaimer I received this product for free but my opinions are my own. That being said this is absolutely the best product ever made.",
        "I was selected as a product tester and I have to say this exceeded all expectations. Truly a premium product in every way.",
        "Full disclosure the company sent me this to try. With that said I genuinely believe this is the best in its category.",
        "I got an early access sample and I am blown away. This will revolutionize the market when it launches. Trust me on this.",
        "The brand reached out to me as an influencer and I am glad they did. This product is genuinely incredible and I stand by that.",

        "DO NOT buy the other brand! THIS is the only one that works! The competition is TERRIBLE! Buy THIS instead!!!",
        "I tried the competitor product and it was AWFUL! This one is a MILLION times better! SWITCH NOW!!!",
        "Other brands are SCAMS! This is the ONLY legitimate product! Trust me I know! BUY THIS ONE!!!",
        "WARNING: Other products in this category are FAKE! Only THIS product delivers REAL results! BUY NOW!!!",
        "Every other product is WORTHLESS compared to this MASTERPIECE! Save yourself time and just BUY THIS ONE!!!",
        "I wasted hundreds of dollars on Brand X before discovering this superior alternative. There is simply no comparison.",
        "Having used the leading competitor for two years I switched to this and the difference is night and day. Far superior.",
        "Brand X broke after a week. Brand Y was overpriced. This product is the only one that actually delivers on its promises.",
        "I returned three competing products before finding this one. Save yourself the trouble and just buy this from the start.",
        "The market is full of inferior products but this one genuinely stands above all of them. Nothing else is worth your money.",

        "HURRY! Buy this before it sells out! BEST product ever! Limited stock! You NEED this! Amazing!!!",
        "ORDER NOW before the price goes up! This INCREDIBLE product is worth TRIPLE the price! MUST HAVE!!!",
        "Don't WAIT! Buy this TODAY! Tomorrow might be too late! BEST purchase you'll ever make! GUARANTEED!!!",
        "SELLING OUT FAST! Get yours before it's GONE! BEST product EVER! You'll THANK me later!!!",
        "LIMITED TIME OFFER! This PERFECT product is available NOW! Don't miss this OPPORTUNITY! 5 STARS!!!",

        "Great product. Works well. Highly recommended. Five stars. Will buy again. Very satisfied with purchase.",
        "Excellent quality. Fast shipping. Perfect condition. As described. Very happy. Recommended seller.",
        "Love it. Great quality. Fast delivery. Will definitely purchase again. Highly recommend to everyone.",
        "Very satisfied. Product is great. Exactly as described. Fast shipping. Would recommend. Five stars.",
        "Perfect. Love it. Great. Amazing quality. Fast shipping. Will buy more. Highly recommended. Best.",
        "Good product good price good shipping good quality good everything. Highly recommend. Five stars.",
        "Super happy with this purchase. Everything is perfect. Quality is amazing. Will order again soon.",
        "Wonderful product. Exceeded expectations. Will tell all my friends. Best purchase this year.",
        "Outstanding quality. Superior product. Fast delivery. No complaints. Will purchase again. Recommended.",
        "Fantastic buy. Premium quality. Arrived quickly. Very impressed. Going to buy another one. Love it.",
        "Absolutely love this product. It works exactly as described and the quality is top notch. Highly recommend to anyone looking for this type of product. Five stars all the way.",
        "This is a must have product. The quality exceeded my expectations and I will definitely be purchasing more. Everyone should try this. You will not be disappointed at all.",
        "Best purchase I have made in a long time. The product is exactly what I was looking for and it works perfectly. Could not be happier with this buy.",
        "Really impressed with this product. The quality is excellent and it arrived sooner than expected. I have already recommended it to several friends and family members.",
        "What a great find! This product does everything it claims and more. I am very pleased with the quality and performance. Will definitely be a repeat customer.",


        "The product works as described. Shipping was fast. Good value for the money. Would recommend to others looking for this type of item.",
        "Decent quality for the price. The stitching could be better but overall I'm satisfied with my purchase. Took 3 days to arrive.",
        "It's okay. Nothing special but gets the job done. The instructions were a bit confusing at first but figured it out eventually.",
        "Good product but the packaging was damaged when it arrived. The item itself was fine though. Customer service was helpful.",
        "I've been using this for about two weeks now. It works well for what I need. Battery life could be longer though.",
        "Not bad. The color was slightly different from the picture but I still like it. Comfortable and fits well.",
        "Works as expected. Setup was straightforward. The manual could use some improvement but YouTube tutorials helped.",
        "Pretty good quality for this price range. I've seen more expensive options that aren't as good. Minor cosmetic defect on mine.",
        "Bought this for my kitchen and it does the job. A bit noisy but effective. Easy to clean which is nice.",
        "Average product. Does what it says but nothing extraordinary. Delivery was on time. Packaging was adequate.",
        "I like it for the most part. The material feels durable. One small issue is the zipper gets stuck sometimes.",
        "Solid purchase. Used it camping last weekend and it held up well. Slightly heavier than expected but manageable.",
        "The product itself is fine but shipping took longer than expected. About 10 days instead of the promised 5.",
        "Good but not great. Fits the description accurately. I would buy again if the price is right.",
        "After a month of use I can say it's reliable. Had one minor issue but resolved it easily. Satisfied overall.",

        "I bought the medium size in blue. It fits true to size and the color matches the photos. The fabric is cotton blend and feels comfortable. After three washes it still looks good.",
        "Installed this in my bathroom last weekend. The mounting hardware was included which was nice. Took about 45 minutes with basic tools. Holds weight well so far.",
        "Using this for my morning commute which is about 30 minutes. Sound quality is decent for the price range around $25. Bass could be better but mids are clear.",
        "Got this for my daughter's birthday. She uses it for school projects. The screen resolution is adequate for basic tasks. Battery lasts about 6 hours with regular use.",
        "Ordered on Monday received on Thursday. The box was slightly dented but the product inside was wrapped in bubble wrap and undamaged. Works as expected for light use.",
        "This is my third kitchen gadget from this brand. Consistent quality across their product line. This particular model has a slightly more ergonomic handle than the previous version.",
        "Replaced my 5 year old model with this updated version. Noticeable improvement in speed and quieter operation. The new interface takes some getting used to though.",
        "Used this on a 3 day hiking trip. Held up well in light rain. The zipper is sturdy and the compartments are well organized. Weighs about 2 pounds empty.",
        "Bought two of these one for home and one for office. Both work identically out of the box. The cord is about 6 feet long which is adequate for my setup.",
        "After researching for about two weeks I settled on this model. It's mid range in terms of price and features. Does 90 percent of what the expensive models do.",
        "Measured my desk before ordering and this fits perfectly in the 24 inch space. The monitor arm adjusts smoothly. Cable management clips are included which is a bonus.",
        "The 500ml capacity is perfect for my needs. Keeps drinks cold for about 8 hours in my experience. The lid seal is tight and I have not had any leaks.",
        "This runs on 4 AA batteries which last about 3 weeks with daily use. The display is backlit which helps in dim conditions. Accuracy seems good compared to my old one.",
        "Planted these seeds in my garden in early April. Germination rate was about 80 percent which is decent. The plants are now about 12 inches tall and healthy looking.",
        "The 2.4ghz connection is stable up to about 30 feet from my laptop. Beyond that it gets spotty. For my small apartment this range is more than sufficient.",

        "Disappointed with this purchase. The quality does not match the price. The material feels cheap and flimsy. Returning it.",
        "Not worth the money. Broke after two weeks of normal use. Very poor build quality. Would not recommend.",
        "The product arrived late and was not as described. The color is wrong and the size is off. Very frustrating experience.",
        "Terrible customer service. The product stopped working after a month and they won't honor the warranty. Avoid this brand.",
        "I regret buying this. It looks nothing like the photos. The material is thin and see through. Complete waste of money.",
        "Product broke on first use. The hinge snapped right off. Clearly poor manufacturing quality. Requesting a refund immediately.",
        "Very misleading product listing. The dimensions listed are incorrect. What arrived is much smaller than expected. Disappointed.",
        "Save your money. This product is overpriced for what you get. Plenty of better options available at half the price.",
        "Had high hopes based on the brand name but this product let me down. Inconsistent performance and cheap materials.",
        "Two stars because at least it arrived on time. But the product itself is subpar. Wobbly construction and poor finish.",
        "Stopped working after exactly one month. Conveniently just past the return window. Feels like planned obsolescence. Never buying again.",
        "The motor burned out after moderate use. Smelled like burning plastic. Potentially dangerous. Reported to the company with no response.",
        "Ordered the black version and received gray. Tried to exchange but was told the color I wanted is out of stock. Frustrating.",
        "The adhesive backing failed within 48 hours. Product fell off the wall and cracked. Very disappointed with the build quality.",
        "Makes an annoying high pitched noise when running. Not mentioned anywhere in the product listing. Would have returned but past the window.",

        "Pros: lightweight and portable. Cons: battery life is only about 4 hours. Overall decent for casual use but not for heavy workloads.",
        "The good: easy to set up and intuitive controls. The bad: makes a humming noise at higher settings. Acceptable for the price point.",
        "What I like: the compact design fits perfectly on my desk. What I don't: the cable management could be better. Satisfied overall.",
        "Strengths: durable construction and water resistant. Weaknesses: limited color options and the strap feels cheap. Would still recommend.",
        "Good things first: fast performance, sleek design, fair price. Now the bad: gets hot during extended use, fan is a bit loud. 3.5 stars.",
        "Love the functionality hate the aesthetics. It does everything I need but looks outdated compared to competitors. Function over form I guess.",
        "Build quality is excellent but the software is clunky. Hardware gets 5 stars software gets 2. Hoping they release an update soon.",
        "Perfect size and weight for travel. Only issue is the charging port is in an awkward position. Minor inconvenience but worth mentioning.",
        "The sound quality is excellent for music playback. However the microphone picks up a lot of background noise during calls. Mixed bag.",
        "Great value for basic features. If you need advanced functionality look elsewhere. For simple everyday tasks this is more than adequate.",

        "Works fine. Good for the price.",
        "Decent product. As described.",
        "Okay quality. Nothing special but functional.",
        "Does the job. Shipped quickly.",
        "Satisfied with purchase. Fair quality.",
        "It's alright. Gets the job done.",
        "Good value. Would buy again maybe.",
        "Not bad for the price point.",
        "Meets expectations. Standard quality.",
        "Fine product. No major complaints.",
        "Three stars. Average at best.",
        "Underwhelming but functional.",
        "Meh. It works I guess.",
        "Could be better could be worse.",
        "Got what I paid for.",

        "Week 1: great performance. Week 3: still going strong. Month 2: minor slowdown but still usable. Overall satisfied with longevity.",
        "First impression was positive. After a month of daily use I can confirm it holds up well. Some minor scratches on the surface.",
        "Bought this six months ago. It still works reliably. The finish has worn slightly but performance is unchanged.",
        "Day one setup was smooth. Day seven I noticed a small quirk with the settings. Two weeks in and I am happy with it.",
        "Three weeks of testing: Day 1-7 everything perfect. Day 8-14 noticed battery degrades faster. Day 15-21 stabilized at about 5 hours.",
        "Purchased in January and now it is April. Three months of regular use and the product continues to perform as expected.",
        "One year update: still working but showing signs of wear. The button has become less responsive and the finish is scratched.",
        "Month 1: loved it. Month 3: still good. Month 6: performance dropped slightly. Month 9: considering a replacement. Decent lifespan.",
        "After six weeks of daily use the product has developed a small rattle. Still functional but slightly annoying. Otherwise no issues.",
        "Originally gave this 5 stars but downgrading to 3 after four months. Performance degradation is noticeable. Disappointing longevity.",

        "Compared to my previous model from a different brand this is a slight upgrade. Better display but the speakers are about the same.",
        "Previously used brand X for 3 years. Switched to this and it's comparable in most ways. Slightly better ergonomics slightly worse battery.",
        "My friend has the more expensive version and honestly the differences are minimal for everyday use. This one offers better value.",
        "After trying 3 different brands I think this one offers the best balance of price and quality. Not the best in any category but solid overall.",
        "Side by side with the competition this holds its own. Similar specs similar performance. The deciding factor was this one being $30 cheaper.",
        "Have used both this and the premium version. For 80 percent of users this budget friendly option will be more than sufficient.",
        "Tested this alongside two competitors. It ranked second in build quality and first in value. Not the absolute best but best for the money.",
        "Switched from the competitor after their latest model disappointed me. This is comparable in quality and about 20 percent cheaper.",
        "Not as good as the premium brands but significantly better than the budget options. Sits comfortably in the mid range.",
        "My colleague uses a different brand and we compared notes. Performance is nearly identical. Choose whichever is on sale.",

        "I have mixed feelings about this product. On one hand the build quality is solid and it looks great. On the other hand the software is buggy and crashes occasionally. I think with a firmware update this could be a really great product but as it stands now it is just okay.",
        "This product is neither great nor terrible. It does what it is supposed to do without any flair. If you need something basic and reliable this will work. If you want something exciting look elsewhere.",
        "Honestly I am on the fence about this. Some days it works perfectly and I love it. Other days it acts up and I want to return it. Inconsistent quality is my main complaint.",
        "The product itself is decent but the customer experience was poor. Shipping was delayed twice and the tracking never updated. Once I actually received it I was satisfied with the product.",
        "I wanted to love this product because the concept is great. The execution falls a bit short though. With some refinement this could be a top seller but right now it is average.",
        "Initially I was disappointed because it did not match my expectations from the photos. After using it for a week though I have come to appreciate its functionality. Growing on me.",
        "This is a perfectly adequate product. Nothing more nothing less. It will not blow your mind but it will not let you down either. Middle of the road in every way.",
        "Some features work great while others feel half baked. The core functionality is solid but the extras feel like afterthoughts. Decent product with room for improvement.",
        "I almost returned this but decided to give it a chance. Glad I did because after the learning curve it has become quite useful. Not intuitive but effective once you figure it out.",
        "For the price I cannot complain too much. There are minor issues here and there but the overall value proposition is fair. You get what you pay for with this one.",

        "Running this on Windows 11 with 16GB RAM. Installation was straightforward. The software uses about 200MB of disk space. Performance is smooth with no noticeable lag during normal operations.",
        "The dimensions are 12x8x4 inches and it weighs 3.2 pounds. Build material appears to be ABS plastic with rubber grips. The power cable is proprietary unfortunately not USB-C.",
        "Tested the waterproof rating by submerging it for 30 minutes in 1 foot of water. No water ingress detected. The IPX7 rating appears to be accurate based on my informal testing.",
        "Power consumption measured at about 45 watts under load using my kill-a-watt meter. Standby power is less than 1 watt. Reasonable efficiency for this product category.",
        "The Wi-Fi card supports both 2.4GHz and 5GHz bands. In my testing the 5GHz connection achieved about 400Mbps throughput at 10 feet from the router. Drops to 200 at 30 feet.",
        "Took apart the unit to check build quality internally. The PCB is well laid out with proper thermal management. Components appear to be from reputable manufacturers. Good internals.",
        "Color temperature measures about 5600K on my calibration tool. Brightness peaks at 350 nits which is adequate for indoor use but struggles in direct sunlight.",
        "The motor runs at approximately 15000 RPM based on the spec sheet. In practice this translates to smooth operation with minimal vibration. Noise level is about 45dB at arm length.",
        "Battery capacity is rated at 5000mAh. In my testing with moderate use screen brightness at 50 percent and WiFi on it lasted about 7.5 hours. Fast charging brings it to 60 percent in 30 minutes.",
        "Thread count is listed as 400 but feels more like 300 to me based on my experience with bedding. Still soft enough for comfortable sleep. Passed the wrinkle test reasonably well.",
    ],
    'label': [
       
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
     
        1,1,1,1,1,1,1,1,1,1,
      
        1,1,1,1,1,1,1,1,1,1,
      
        1,1,1,1,1,
       
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
      
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
 
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    
        0,0,0,0,0,0,0,0,0,0,
 
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

        0,0,0,0,0,0,0,0,0,0,

        0,0,0,0,0,0,0,0,0,0,

        0,0,0,0,0,0,0,0,0,0,
 
        0,0,0,0,0,0,0,0,0,0,
    ]
}
df = pd.DataFrame(data)
assert len(df[df['label']==1]) + len(df[df['label']==0]) == len(df), "Label count mismatch!"
print(f"\n Dataset loaded: {len(df)} reviews")
print(f"   - Fake reviews: {sum(df['label']==1)}")
print(f"   - Real reviews: {sum(df['label']==0)}")


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def extract_features(text):
    features = {}
    

    features['char_count'] = len(text)
    words = text.split()
    features['word_count'] = len(words)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    

    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['period_count'] = text.count('.')
    features['excl_per_word'] = features['exclamation_count'] / (features['word_count'] + 1)
    
    
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / (len(text) + 1)
    features['all_caps_words'] = sum(1 for w in words if w.isupper() and len(w) > 1)
    features['caps_word_ratio'] = features['all_caps_words'] / (features['word_count'] + 1)
    
    lower_words = [w.lower() for w in words]
    unique_words = set(lower_words)
    features['unique_word_ratio'] = len(unique_words) / (len(lower_words) + 1)
   
    if lower_words:
        word_freq = pd.Series(lower_words).value_counts()
        features['max_word_repeat'] = word_freq.max()
        features['words_repeated_3plus'] = sum(1 for c in word_freq if c >= 3)
    else:
        features['max_word_repeat'] = 0
        features['words_repeated_3plus'] = 0
    
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    features['sentence_count'] = max(len(sentences), 1)
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
    
    blob = TextBlob(text)
    features['polarity'] = blob.sentiment.polarity          
    features['subjectivity'] = blob.sentiment.subjectivity   
    features['extreme_polarity'] = abs(blob.sentiment.polarity)
    fake_words = [
        'amazing', 'perfect', 'best', 'incredible', 'must buy',
        'love', 'wow', 'fantastic', 'miraculous', 'buy now',
        'changed my life', 'ever', 'greatest', 'unbelievable',
        'life changing', 'must have', 'phenomenal', 'flawless',
        'revolutionary', 'miracle', 'superb', 'outstanding',
        'magnificent', 'divine', 'heaven', 'obsessed',
        'game changer', 'hurry', 'limited', 'act now',
        'selling fast', 'last chance', 'urgent', 'guaranteed',
        'don\'t wait', 'buy buy', 'trust me', 'believe me'
    ]
    text_lower = text.lower()
    features['fake_word_count'] = sum(1 for w in fake_words if w in text_lower)
    features['fake_word_ratio'] = features['fake_word_count'] / (features['word_count'] + 1)
    
    marketing_words = [
        'innovative', 'premium', 'seamless', 'streamline', 'optimize',
        'cutting edge', 'state of the art', 'world class', 'best in class',
        'unparalleled', 'second to none', 'pinnacle', 'epitome',
        'meticulously', 'craftsmanship', 'transformative', 'revolutionize',
        'indispensable', 'instrumental', 'exceptional', 'superior',
        'unprecedented', 'remarkable', 'extraordinary', 'sophistication',
        'refinement', 'excellence', 'masterfully', 'thoughtful design',
        'stands in a class', 'sets a new standard', 'quantum leap'
    ]
    features['marketing_word_count'] = sum(1 for w in marketing_words if w in text_lower)
    
    
    specific_indicators = [
        'inch', 'feet', 'pound', 'ounce', 'hour', 'minute', 'day', 'week',
        'month', 'year', 'dollar', '$', 'percent', '%', 'model', 'version',
        'size', 'color', 'weight', 'battery', 'shipped', 'arrived', 'returned',
        'warranty', 'refund', 'customer service', 'delivery', 'packaging'
    ]
    features['specificity_count'] = sum(1 for w in specific_indicators if w in text_lower)
    
   
    features['number_count'] = len(re.findall(r'\d+', text))
    
  
    hedging_words = [
        'maybe', 'perhaps', 'somewhat', 'slightly', 'a bit', 'kind of',
        'sort of', 'fairly', 'reasonably', 'adequate', 'decent', 'okay',
        'alright', 'not bad', 'could be better', 'minor', 'small issue',
        'however', 'although', 'but', 'though', 'despite', 'except'
    ]
    features['hedging_count'] = sum(1 for w in hedging_words if w in text_lower)
    
    return features

print("")
df['processed'] = df['review'].apply(preprocess_text)
feature_df = df['review'].apply(lambda x: pd.Series(extract_features(x)))
print(f"   Features extracted: {list(feature_df.columns)}")
X_text = df['processed']
y = df['label']

X_train_text, X_test_text, X_train_feat, X_test_feat, y_train, y_test = train_test_split(
    X_text, feature_df, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training: {len(X_train_text)} | Testing: {len(X_test_text)}")

print("")
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)
X_train_combined = hstack([X_train_tfidf, X_train_feat.values])
X_test_combined = hstack([X_test_tfidf, X_test_feat.values])

print(f"features: {X_train_combined.shape[1]}")

lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced')
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42, learning_rate=0.1)

model = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
    voting='soft'
)
model.fit(X_train_combined, y_train)

y_pred = model.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'=' * 60}")
print(f"ACCURACY: {accuracy * 100:.1f}%")
print(f"{'=' * 60}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

test_reviews = [
    "BEST PRODUCT EVER!!! BUY NOW!!! AMAZING!!!",
    "It's okay. Works fine for the price. Nothing special.",
    "At its core this product is designed for maximum efficiency. I have found it to be instrumental in streamlining my morning routine.",
    "Bought this 3 months ago. The 2.5 inch screen is clear. Battery lasts about 6 hours. Decent for $30.",
    "This product truly revolutionizes the way you think about everyday tasks. The engineering behind it is simply remarkable.",
    "Not great. The zipper broke after a week. Returning it for a refund.",
    "I received this product for free and it is absolutely the best thing ever made! Everyone must buy this!",
    "Does what it says. Nothing more nothing less. Adequate for daily use.",
]

for rev in test_reviews:
    proc = preprocess_text(rev)
    tfidf_feat = tfidf.transform([proc])
    num_feat = pd.DataFrame([extract_features(rev)])
    combined = hstack([tfidf_feat, num_feat.values])
    pred = model.predict(combined)[0]
    prob = model.predict_proba(combined)[0]
    label = "FAKE" if pred == 1 else "REAL"
    conf = max(prob) * 100
    short = rev[:70] + "..." if len(rev) > 70 else rev
    print(f"   {label} ({conf:.0f}%) → \"{short}\"")

print("\n💾 Saving enhanced model...")
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/fake_review_model.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')