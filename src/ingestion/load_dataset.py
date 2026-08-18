import os
from typing import List, Dict, Any
from datasets import load_dataset

def fetch_msmarco_passages(limit: int = 500) -> List[Dict[str, Any]]:
    print("[*] Loading Master Knowledge Base for India & AI4Bharat MSMARCO-XI...")
    documents = []

    master_india_knowledge = [
        # =========================================================================
        # 1. IMPORTANT DATES, INDEPENDENCE & CONSTITUTION
        # =========================================================================
        {"q": "when did india get independence day date", "text": "India gained independence from British colonial rule on August 15, 1947."},
        {"q": "when is republic day celebrated in india date", "text": "Republic Day is celebrated across India on January 26 every year, commemorating the date the Constitution of India came into effect in 1950."},
        {"q": "when was the constitution of india adopted and enacted", "text": "The Constitution of India was adopted by the Constituent Assembly on November 26, 1949 (celebrated as Constitution Day) and came into legal effect on January 26, 1950."},
        {"q": "when was goa liberated operation vijay date", "text": "Goa was liberated from 451 years of Portuguese colonial rule on December 19, 1961 during Operation Vijay and integrated into the Indian Union."},
        {"q": "when is mahatma gandhi birthday gandhi jayanti", "text": "Gandhi Jayanti is celebrated on October 2 every year to mark the birth anniversary of Mahatma Gandhi, observed globally as the International Day of Non-Violence."},
        {"q": "when was national flag of india adopted", "text": "The National Flag of India (Tiranga) was adopted by the Constituent Assembly on July 22, 1947, designed by Pingali Venkayya."},
        {"q": "when did kargil war happen", "text": "The Kargil War took place between May and July 1999 in Jammu and Kashmir's Kargil district, with India declaring victory on Kargil Vijay Diwas on July 26, 1999."},

        # =========================================================================
        # 2. MAJOR FESTIVALS & CELEBRATIONS OF INDIA
        # =========================================================================
        {"q": "what is diwali festival of lights", "text": "Diwali (Deepavali) is the Hindu festival of lights, celebrating the triumph of light over darkness and the return of Lord Rama to Ayodhya."},
        {"q": "what is holi festival of colours", "text": "Holi is the ancient Hindu festival of colours, celebrating the arrival of spring, love, and the victory of good over evil (Holika Dahan)."},
        {"q": "what is onam festival of kerala", "text": "Onam is the official annual harvest festival and cultural celebration of Kerala, commemorating the mythical King Mahabali."},
        {"q": "what is pongal festival of tamil nadu", "text": "Pongal is a four-day harvest festival celebrated by Tamils in Tamil Nadu and Puducherry in mid-January, dedicated to the Sun God Surya."},
        {"q": "what is bihu festival of assam", "text": "Bihu is the chief cultural and harvest festival of Assam, celebrated in three seasons: Rongali (Bohag) Bihu, Kongali (Kati) Bihu, and Bhogali (Magh) Bihu."},
        {"q": "what is durga puja of west bengal kolkata", "text": "Durga Puja is the largest annual Hindu festival in West Bengal, celebrating Goddess Durga's victory over the demon Mahishasura. It is inscribed on UNESCO's Intangible Cultural Heritage list."},
        {"q": "what is ganesh chaturthi festival of maharashtra", "text": "Ganesh Chaturthi is a major ten-day festival celebrated with grand public pandals and idol immersions, widely celebrated across Maharashtra."},
        {"q": "what is chath puja festival of bihar", "text": "Chhath Puja is an ancient Hindu Vedic festival celebrated mainly in Bihar, Jharkhand, and eastern UP, dedicated to the Sun God Surya and Chhathi Maiya."},
        {"q": "what is baisakhi festival of punjab", "text": "Baisakhi (Vaisakhi) is the spring harvest festival of Punjab, marking the formation of the Khalsa Panth by Guru Gobind Singh in 1699."},
        {"q": "what is hornbill festival of nagaland", "text": "The Hornbill Festival, called the 'Festival of Festivals', is an annual cultural celebration held in Kisama, Nagaland, showcasing the traditions of all Naga tribes."},
        {"q": "what is pushkar camel fair rajasthan", "text": "The Pushkar Camel Fair is an annual multi-day livestock fair and cultural festival held in Pushkar, Rajasthan, on the banks of Pushkar Lake."},

        # =========================================================================
        # 3. NATIONAL SYMBOLS, MOTTO & EMBLEMS
        # =========================================================================
        {"q": "what is national animal of india", "text": "The Bengal Tiger (Panthera tigris) is the national animal of India, symbolizing strength and agility."},
        {"q": "what is national bird of india", "text": "The Indian Peacock (Pavo cristatus) is the national bird of India."},
        {"q": "what is national flower of india", "text": "The Lotus (Nelumbo nucifera) is the sacred national flower of India."},
        {"q": "what is national fruit of india", "text": "Mango (Mangifera indica) is the national fruit of India, known as the King of Fruits."},
        {"q": "what is national tree of india", "text": "The Banyan Tree (Ficus benghalensis) is the national tree of India."},
        {"q": "what is national river of india", "text": "The Ganga (Ganges) is the national river of India, originating from the Gangotri Glacier."},
        {"q": "what is national aquatic animal of india", "text": "The Ganges River Dolphin (Platanista gangetica) is the national aquatic animal of India."},
        {"q": "what is national emblem and motto of india", "text": "The Lion Capital of Ashoka at Sarnath is the national emblem of India, inscribed with the motto 'Satyameva Jayate' (Truth Alone Triumphs) in Devanagari."},
        {"q": "who composed national anthem and song of india", "text": "Jana Gana Mana is the national anthem of India composed by Rabindranath Tagore. Vande Mataram is the national song composed by Bankim Chandra Chatterjee."},

        # =========================================================================
        # 4. LEADERSHIP, CONSTITUTION & GOVERNMENT
        # =========================================================================
        {"q": "who is current president of india", "text": "Droupadi Murmu is the 15th and current President of India."},
        {"q": "who is current prime minister of india", "text": "Narendra Modi is the current Prime Minister of India."},
        {"q": "who was first president of india", "text": "Dr. Rajendra Prasad was the first President of independent India from 1950 to 1962."},
        {"q": "who was first prime minister of india", "text": "Jawaharlal Nehru was the first Prime Minister of independent India from 1947 to 1964."},
        {"q": "who is father of indian constitution", "text": "Dr. B. R. Ambedkar is recognized as the chief architect and Father of the Constitution of India."},
        {"q": "who is iron man of india sardar patel", "text": "Sardar Vallabhbhai Patel is known as the Iron Man of India and Bismarck of India for unifying 565 princely states into the Indian Union."},
        {"q": "what is capital of india", "text": "New Delhi is the official national capital of India and the seat of the Executive, Legislative, and Judiciary."},

        # =========================================================================
        # 5. ALL 28 STATES (CAPITALS & OFFICIAL LANGUAGES)
        # =========================================================================
        {"q": "capital and language of andhra pradesh", "text": "Amaravati is the capital of Andhra Pradesh. The official language is Telugu."},
        {"q": "capital and language of arunachal pradesh", "text": "Itanagar is the capital of Arunachal Pradesh. The official language is English."},
        {"q": "capital and language of assam", "text": "Dispur is the capital of Assam. The official language is Assamese."},
        {"q": "capital and language of bihar", "text": "Patna is the capital of Bihar. The official languages are Hindi and Urdu."},
        {"q": "capital and language of chhattisgarh", "text": "Raipur is the capital of Chhattisgarh. The official language is Hindi and Chhattisgarhi."},
        {"q": "capital and language of goa", "text": "Panaji is the capital of Goa. The official language is Konkani."},
        {"q": "capital and language of gujarat", "text": "Gandhinagar is the capital of Gujarat. The official language is Gujarati."},
        {"q": "capital and language of haryana", "text": "Chandigarh is the capital of Haryana. The official language is Hindi."},
        {"q": "capital and language of himachal pradesh", "text": "Shimla (summer) and Dharamshala (winter) are the capitals of Himachal Pradesh. The official language is Hindi."},
        {"q": "capital and language of jharkhand", "text": "Ranchi is the capital of Jharkhand. The official language is Hindi."},
        {"q": "capital and language of karnataka", "text": "Bengaluru (Bangalore) is the capital of Karnataka. The official language is Kannada."},
        {"q": "capital and language of kerala", "text": "Thiruvananthapuram is the capital of Kerala. The official language is Malayalam."},
        {"q": "capital and language of madhya pradesh", "text": "Bhopal is the capital of Madhya Pradesh. The official language is Hindi."},
        {"q": "capital and language of maharashtra", "text": "Mumbai is the capital of Maharashtra. The official language is Marathi."},
        {"q": "capital and language of manipur", "text": "Imphal is the capital of Manipur. The official language is Meitei (Manipuri)."},
        {"q": "capital and language of meghalaya", "text": "Shillong is the capital of Meghalaya. The official language is English."},
        {"q": "capital and language of mizoram", "text": "Aizawl is the capital of Mizoram. The official language is Mizo and English."},
        {"q": "capital and language of nagaland", "text": "Kohima is the capital of Nagaland. The official language is English."},
        {"q": "capital and language of odisha", "text": "Bhubaneswar is the capital of Odisha. The official language is Odia."},
        {"q": "capital and language of punjab", "text": "Chandigarh is the capital of Punjab. The official language is Punjabi."},
        {"q": "capital and language of rajasthan", "text": "Jaipur is the capital of Rajasthan. The official language is Hindi."},
        {"q": "capital and language of sikkim", "text": "Gangtok is the capital of Sikkim. The official languages are English, Nepali, Sikkimese, and Lepcha."},
        {"q": "capital and language of tamil nadu", "text": "Chennai is the capital of Tamil Nadu. The official language is Tamil."},
        {"q": "capital and language of telangana", "text": "Hyderabad is the capital of Telangana. The official languages are Telugu and Urdu."},
        {"q": "capital and language of tripura", "text": "Agartala is the capital of Tripura. The official languages are Bengali, English, and Kokborok."},
        {"q": "capital and language of uttar pradesh up", "text": "Lucknow is the capital of Uttar Pradesh. The official language of Uttar Pradesh is Hindi, with Urdu as the second official language."},
        {"q": "capital and language of uttarakhand", "text": "Dehradun (winter) and Gairsain (summer) are the capitals of Uttarakhand. The official language is Hindi."},
        {"q": "capital and language of west bengal", "text": "Kolkata is the capital of West Bengal. The official language is Bengali."},

        # =========================================================================
        # 6. UNION TERRITORIES (CAPITALS & LANGUAGES)
        # =========================================================================
        {"q": "capital and language of jammu and kashmir", "text": "Srinagar (summer) and Jammu (winter) are the capitals of Jammu and Kashmir. Official languages are Kashmiri, Dogri, Hindi, Urdu, and English."},
        {"q": "capital and language of ladakh", "text": "Leh and Kargil are the capitals of Ladakh. Official languages are Hindi and English."},
        {"q": "capital and language of delhi", "text": "New Delhi is the capital of Delhi NCT and India. Official languages are Hindi and English."},
        {"q": "capital and language of chandigarh", "text": "Chandigarh is a Union Territory and joint capital of Punjab and Haryana."},
        {"q": "capital and language of puducherry", "text": "Puducherry is the capital of Puducherry UT. Official languages are Tamil, French, and English."},
        {"q": "capital and language of andaman and nicobar", "text": "Port Blair is the capital of Andaman and Nicobar Islands."},
        {"q": "capital and language of lakshadweep", "text": "Kavaratti is the capital of Lakshadweep. The official language is Malayalam and English."},
        {"q": "capital and language of dadra and nagar haveli daman diu", "text": "Daman is the capital of Dadra and Nagar Haveli and Daman and Diu."},

        # =========================================================================
        # 7. FAMOUS MONUMENTS & HERITAGE SITES
        # =========================================================================
        {"q": "taj mahal location who built", "text": "The Taj Mahal is an ivory-white marble mausoleum in Agra, Uttar Pradesh, built by Mughal Emperor Shah Jahan in memory of Mumtaz Mahal. It is a UNESCO World Heritage site."},
        {"q": "red fort location lal qila", "text": "The Red Fort (Lal Qila) is a historic fort in Old Delhi, built by Shah Jahan, where the Prime Minister hoists the National Flag on Independence Day."},
        {"q": "qutub minar location height", "text": "Qutub Minar is a 72.5-metre tall minaret in Mehrauli, New Delhi, founded by Qutb-ud-din Aibak and completed by Iltutmish."},
        {"q": "statue of unity tallest statue location", "text": "The Statue of Unity in Kevadia (Ekta Nagar), Gujarat, is the world's tallest statue (182 metres), honoring Sardar Vallabhbhai Patel."},
        {"q": "konark sun temple location", "text": "The Konark Sun Temple, also known as the Black Pagoda, is a 13th-century CE Sun temple in Konark, Odisha, built by King Narasimhadeva I."},
        {"q": "hawa mahal location pink city", "text": "Hawa Mahal (Palace of Winds) is a red and pink sandstone palace in Jaipur, Rajasthan, built in 1799 by Maharaja Sawai Pratap Singh."},
        {"q": "gateway of india location mumbai", "text": "The Gateway of India is an arch-monument located on the waterfront at Apollo Bunder in Mumbai, Maharashtra."},
        {"q": "india gate location new delhi", "text": "India Gate is a war memorial located along the Rajpath (Kartavya Path) in New Delhi, honoring soldiers of the British Indian Army who died in World War I."},

        # =========================================================================
        # 8. SPACE, SCIENCE & DEFENCE (ISRO / DRDO)
        # =========================================================================
        {"q": "what is isro indian space research organisation", "text": "ISRO (Indian Space Research Organisation) is the national space agency of India, headquartered in Bengaluru, Karnataka."},
        {"q": "what is chandrayaan 3 lunar mission isro", "text": "Chandrayaan-3 was India's historic lunar mission that successfully soft-landed near the Moon's south pole on August 23, 2023 (National Space Day)."},
        {"q": "what is mangalyaan mars orbiter mission", "text": "Mangalyaan (Mars Orbiter Mission / MOM) was ISRO's first interplanetary mission, making India the first Asian nation to reach Martian orbit in its maiden attempt in 2014."},
        {"q": "what is aditya l1 solar mission", "text": "Aditya-L1 is India's first dedicated space observatory to study the Sun, placed in a halo orbit around the Sun-Earth L1 Lagrange point."},
        {"q": "who was missile man of india apj abdul kalam", "text": "Dr. A. P. J. Abdul Kalam was an Indian aerospace scientist, the 11th President of India (2002-2007), and widely known as the 'Missile Man of India'."},

        # =========================================================================
        # 9. GEOGRAPHY, RIVERS & PEAKS
        # =========================================================================
        {"q": "longest river in india ganga", "text": "The Ganga is the longest river in India (2,525 km), originating from the Gangotri Glacier at Gomukh and flowing into the Bay of Bengal."},
        {"q": "highest mountain peak in india kangchenjunga", "text": "Kangchenjunga (8,586 metres) in Sikkim along the India-Nepal border is the highest mountain peak in India and third highest in the world."},
        {"q": "largest desert in india thar desert", "text": "The Thar Desert (Great Indian Desert) is the largest arid region in India, spanning across Rajasthan, Gujarat, Punjab, and Haryana."},
        {"q": "largest freshwater lake in india wular lake", "text": "Wular Lake in Jammu and Kashmir is the largest freshwater lake in India, fed by the Jhelum River."},
        {"q": "largest saltwater lake in india chilika lake", "text": "Chilika Lake in Odisha is the largest brackish water lagoon in India and second largest coastal lagoon in the world."}
    ]

    for idx, item in enumerate(master_india_knowledge):
        documents.append({
            "doc_id": f"master_india_{idx}",
            "query": item["q"],
            "passages": [item["text"]],
            "language": "en"
        })

    # Optional streaming from AI4Bharat MSMARCO-XI
    try:
        ds = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train", streaming=True)
        for idx, item in enumerate(ds):
            if len(documents) >= limit:
                break
            passages = item.get("passages", {}).get("English_passages", [])
            valid_p = [p.strip() for p in passages if p and len(p.strip()) > 30]
            if valid_p:
                documents.append({
                    "doc_id": str(item.get("query_id", f"online_{idx}")),
                    "query": item.get("Eng_Query", item.get("query", "")),
                    "passages": valid_p,
                    "language": "en"
                })
    except Exception:
        pass

    print(f"[OK] Indexed {len(documents)} master India factual passages.")
    return documents