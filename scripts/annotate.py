import re, spacy
from spacy.tokens import Span #type: ignore

# List of currency symbols and three-letter codes
CURRENCY_SYMBOLS = {"$", "¥", "£", "€", "kr", "₽", "R$", "₹", "Rp", "₪", "zł", "Rs", "₺", "RS"}

CURRENCY_CODES = {"USD", "EUR", "CNY", "JPY", "GBP", "NOK", "DKK", "CAD", "RUB", "MXN", "ARS", "BGN",
                  "BRL", "CHF", "CLP", "CZK", "INR", "IDR", "ILS", "IRR", "IQD", "KRW", "KZT", "NGN",
                  "QAR", "SEK", "SYP", "TRY", "UAH", "AED", "AUD", "COP", "MYR", "SGD", "NZD", "THB",
                  "HUF", "HKD", "ZAR", "PHP", "KES", "EGP", "PKR", "PLN", "XAU", "VND", "GBX"}

# Full list of country names
COUNTRIES = {'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua', 'Argentina',
             'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
             'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 
             'Bosnia Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina', 'Burundi',
             'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Central African Republic', 'Chad', 
             'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Costa Rica', 'Croatia', 'Cuba', 
             'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 
             'East Timor', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 
             'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 
             'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 
             'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 
             'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 
             'Kazakhstan', 'Kenya', 'Kiribati', 'Korea North', 'Korea South', 'Kosovo', 'Kuwait', 
             'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 
             'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar', 'Malawi', 
             'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 
             'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 
             'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 
             'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Palau', 
             'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 
             'Qatar', 'Romania', 'Russian Federation', 'Rwanda', 'St Kitts & Nevis', 'St Lucia', 
             'Saint Vincent & the Grenadines', 'Samoa', 'San Marino', 'Sao Tome & Principe', 
             'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 
             'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan', 
             'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 
             'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Togo', 'Tonga', 
             'Trinidad & Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 
             'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 
             'Uzbekistan', 'Vanuatu', 'Vatican City', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 
             'Zimbabwe', 'USA', 'UK', 'Russia', 'South Korea'}

MONTHS = {"January", "February", "March", "April", "May", "June", "July", 
          "August", "September", "October", "November", "December"}
MONTHS_ABBRV = {"Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", 
                "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec."}

NOT_NAMED_ENTITIES = ["Government", "Applicant", "Registry", "Court", "Commission", "Chamber", "Grand Chamber", 
"Rules of Court", "Convention", "European Commission of Human Rights", "Section", "States", "Committee",
"Registrar", "State", "Board", "Convention for the Protection of Human Rights and Fundamental Freedoms", "day",
"appointed day", "few hours", "next day", "previous day", "previous days", "next days", "evening", "morning", "afternoon",
"night", "month", "months", "same day", "monthly", "daily", "yearly", "weekly", "several days", "several hours",
"several weeks", "several months", "many days", "many hours", "many weeks", "many months", "judge", "solicitor",
"solicitors", "today", "Agent", "Legal Adviser", "Ministry for Foreign Affairs","Ministry of Foreign Affairs",
"Ministry of Justice", "European Community", "First Section", "Second Section", "Third Section",
"Fourth Section", "Fifth Section", "Sixth Section", "Seventh Section", "Delegate", "Councel", 
"Agent of the Government", "Delegate of the Commission", "European Convention on Human Rights", "Assembly",
"Court of Appeal", "European Court of Human Rights", "European Court of Human Right", "European", "Sections",
"1 November 2001", "inter alia", "1 November 1998", "1 November 2004", "Fourth Section of the Court"]

class Annotator:

    def __init__(self, spacy_model="en_core_web_md"):
        self.nlp = spacy.load(spacy_model)

    def annotate_all(self, texts):
        for doc in self.nlp.pipe(texts):
            yield self._annotate(doc)
    
    def annotate(self, text):
        doc = self.nlp(text)
        return self._annotate(doc)
            
    def _annotate(self, doc):

        doc = remove_errors(doc)
        doc = _correct_entities(doc)
        post_ents = []
        for ent in doc.ents:
            if ent.label_=="PERSON":
                start = ent.start
                if start > 0 and doc[start-1].text.lower() in {"mr", "mrs", "ms", "mr.", "mrs.", "ms.",
                "dr", "dr.", "sir", "prof", "miss", "judge", "justice"}:
                    start = start-1
                post_ents.append((start, ent.end, ent.label_))
            elif ent.label_=="GPE":
                if "Kingdom of" in ent.text or "Republic of" in ent.text:
                    print("yes here", ent)
                    post_ents.append((ent.start, ent.end, "ORG"))                
                elif ent.text in COUNTRIES:
                    preceding_tokens = {tok.text for tok in doc[max(0, ent.start-2):ent.start]}
                    if set.intersection(preceding_tokens, {"in", "from"}):
                        post_ents.append((ent.start, ent.end, "LOC"))
                    else:
                        post_ents.append((ent.start, ent.end, "ORG"))
                else:
                    post_ents.append((ent.start, ent.end, "LOC"))
            elif ent.label_ in {"LANGUAGE", "NORP"}: 
                post_ents.append((ent.start, ent.end, "DEM"))
            elif ent.label_ in {"LOC", "FAC"}:
                post_ents.append((ent.start, ent.end, "LOC"))
            elif ent.label_ in {"PRODUCT", "EVENT", "WORK_OF_ART"}:
                post_ents.append((ent.start, ent.end, "MISC"))
            elif ent.label_ in "ORG":
                post_ents.append((ent.start, ent.end, "ORG"))
            elif ent.label_ in {"DATE", "TIME"}:
                if re.match("\d+$", ent.text) and len(ent.text) < 4:
                    continue
    #            elif "born" in [tok.lower_ for tok in doc[max(0,ent.start-3):ent.start]]:
    #                post_ents.append((ent.start, ent.end, "DEM"))
                else:
                    post_ents.append((ent.start, ent.end, "DATETIME"))
            elif ent.label_ in {"PERCENT", "QUANTITY", "MONEY"}:
                post_ents.append((ent.start, ent.end, "QUANTITY"))
            elif ent.label_ in {"CARDINAL"} and len(ent.text) > 4 and re.search("\d", ent.text):
                post_ents.append((ent.start, ent.end, "CODE"))
        for tok in doc:
            if re.match("\d{4,7}\/\d{1,3}$", tok.text):
                post_ents.append((tok.i, tok.i+1, "CODE"))

        post_ents = fix_overlaps(post_ents, doc)
        doc.ents = [Span(doc, start, end, label=label) for (start, end, label) in post_ents]
        doc = _correct_entities(doc)

        return doc
    
def _correct_entities(doc, recursive=True):
    """Correct the named entities in Spacy documents (wrong boundaries or entity type)"""

    new_ents = []
    has_changed = False
    
    # Remove errors (words or phrases that are never named entities)
    existing_ents = [ent for ent in doc.ents if ent.text not in NOT_NAMED_ENTITIES]
    if len(existing_ents) < len(doc.ents):
        has_changed = True

    for ent in existing_ents:
        # If the token after the span is a currency symbol, extend the span on the right side
        if (ent.end < len(doc) and (doc[ent.end].lemma_.lower() in (CURRENCY_SYMBOLS | {"euro", "cent", "ruble"})
                                       or doc[ent.end].text.upper() in CURRENCY_CODES)
            and ((ent.end == len(doc)-1) or (doc[ent.end].ent_type==0))):
            new_ents.append((ent.start, ent.end+1, spacy.symbols.MONEY)) #type: ignore
            has_changed = True

        # Correct entities that go one token too far and include the prepositions to or as
        if (doc[ent.end-1].text in {"to", "as"}):
            new_ents.append((ent.start, ent.end-1, ent.label))
            has_changed=True
          
                   
        # Correct entities with a genitive marker
        if (doc[ent.end-1].text=="'s" and ent.label==spacy.symbols.PERSON):  #type: ignore
            new_ents.append((ent.start, ent.end-1, spacy.symbols.PERSON))  #type: ignore
            has_changed=True
            
        # Extend MONEY spans if the following token is "million", "billion", etc.
        elif (ent.end < len(doc) and doc[ent.end].lemma_.lower() in {"million", "billion", "mln", "bln", "bn", "thousand", 
                                                                           "m", "k", "b", "m.", "k.", "b.", "mln.", "bln.", "bn."}
            and ent.label in {spacy.symbols.MONEY, spacy.symbols.CARDINAL}):  #type: ignore
            new_ents.append((ent.start, ent.end+1, ent.label))
            has_changed = True
            
        # If the token preceding the span is a currency symbol or code, expend the span on the left
        elif (ent.start > 0 and doc[ent.start-1].ent_type==0 and 
              (doc[ent.start-1].text in CURRENCY_SYMBOLS or doc[ent.start-1].text in CURRENCY_CODES)):
            new_ents.append((ent.start-1, ent.end, spacy.symbols.MONEY))  #type: ignore
            has_changed = True
            
        # If the first token of the span is a currency symbol, make sure its label is MONEY
        elif (len(doc[ent.start].text) >=3 and doc[ent.start].text[:3] in CURRENCY_CODES 
              and ent.label != spacy.symbols.MONEY):  #type: ignore
            new_ents.append((ent.start, ent.end, spacy.symbols.MONEY))     #type: ignore 
            has_changed = True
        
        # If the entity contains "per cent", make sure the entity label is percent, not money
        elif len(ent)>=3 and ent.text.endswith("per cent") and ent.label != spacy.symbols.PERCENT :  #type: ignore
            new_ents.append((ent.start, ent.end, spacy.symbols.PERCENT))      #type: ignore
            has_changed = True 
        
        # Fix expression with pennies such as 520.0p
        elif doc[ent.end-1].text[0].isdigit() and ent.text[-1]=='p' and ent.label != spacy.symbols.MONEY:  #type: ignore
            new_ents.append((ent.start, ent.end, spacy.symbols.MONEY))      #type: ignore
            has_changed = True

        # Corrections for dates
        elif (ent.start > 0 and re.match("\d\d?$", doc[ent.start-1].text) and doc[ent.start].text in MONTHS|MONTHS_ABBRV):
            new_ents.append((ent.start-1, ent.end, spacy.symbols.DATE))    #type: ignore  
            has_changed = True   

        # Corrections for countries
        elif (ent.start > 1 and (doc[ent.start-2:ent.start].text=="Kingdom of" or doc[ent.start-2:ent.start].text=="Republic of")):
            new_ents.append((ent.start-2, ent.end, spacy.symbols.ORG))      #type: ignore
            has_changed = True  

        elif ent.end < len(doc) and len(ent) > 1 and " (" in ent.text:
            for i in range(ent.start+1, ent.end):
                if doc[i].text.startswith("("):
                    new_ents.append((ent.start, i, ent.label))
                    has_changed = True
                    break
            
        # Otherwise, add the entity if it does not overlap with any of the new ones
        elif not new_ents or new_ents[-1][1] < ent.end:
            new_ents.append((ent.start, ent.end, ent.label))
        
    # Loop on the tokens to find occurrences of currency symbols followed by numeric values
    # (which remain often undetected in the current entities)
    for token in doc:
        if (token.text in CURRENCY_CODES|CURRENCY_SYMBOLS and token.ent_type!=spacy.symbols.MONEY   #type: ignore
            and token.i < len(doc)-1 and (doc[token.i+1].text[0].isdigit() or 
                                             doc[token.i+1].text in CURRENCY_SYMBOLS)):
            entity_end = token.i+2
            for i in range(token.i+2, len(doc)):
                if any([i>=start and i <end for start,end, _ in new_ents]):
                    entity_end = i+1
                else:
                    break
            new_ents.append((token.i,entity_end, spacy.symbols.MONEY))  #type: ignore
            has_changed = True

    nb_ents = len(new_ents)
    new_ents = fix_overlaps(new_ents, doc)
    if len(new_ents) != nb_ents:
        has_changed = True

    for i, (start, end, label) in enumerate(new_ents):
        if doc[start].lower_ in {"the", "a"}:
            new_ents[i] = (start+1, end, label)
            has_changed = True

    
    # If something has changed, create new spans and run the method once more
    if has_changed:
        new_spans = tuple(spacy.tokens.Span(doc, start, end, symbol)   #type: ignore
                          for (start, end, symbol) in new_ents)
        doc.ents = new_spans
        if recursive:  
            return _correct_entities(doc, False)
    return doc


def fix_overlaps(new_ents, doc):
    new_ents = sorted(new_ents, key=lambda p: p[0])

    # We need to deal with overlapping named entities by merging them
    merge_loop = True
    while merge_loop:
        merge_loop = False
        new_ents2 = list(new_ents)
        for i, (ent_start, ent_end, ent_label) in enumerate(new_ents2):
            for j, (ent2_start, ent2_end, ent2_label) in enumerate(new_ents2[i+1:i+5]):

                if ent_end>ent2_start or (ent_end==ent2_start and ent_label==ent2_label):
                    del new_ents[i+j]
                    # If one label is MONEY, assume the merge is MONEY as well
                    if ent_label in {"MONEY", spacy.symbols.MONEY} or ent2_label in {"MONEY", spacy.symbols.MONEY}:  #type: ignore
                        new_ents[i] = (ent_start, ent2_end, ent_label)
                    # Otherwise, take the label of the longest sequence
                    elif ent2_end-ent2_start >= ent_end-ent_start:
                        new_ents[i] = (ent_start, ent2_end, ent2_label)
                    else:
                        new_ents[i] = (ent_start, ent2_end, ent_label)
                    merge_loop = True
                    break

                # If two GPEs are separated with a comma
                elif (ent_label in {"LOC", "GPE", spacy.symbols.LOC, spacy.symbols.GPE} and   #type: ignore
                      ent2_label in {"LOC", "GPE", spacy.symbols.LOC, spacy.symbols.GPE} and   #type: ignore
                      ent_end + 1 == ent2_start and doc[ent_end].text==","):
                    del new_ents[i+j]
                    new_ents[i] = (ent_start, ent2_end, ent_label)
                    merge_loop = True
                    break

            if merge_loop:
                break
    return new_ents

def remove_errors(doc):
    post_ents = []
    for ent in doc.ents:
        before_space = doc[ent.start:ent.start+2].text.split()[0]
        if ent.start==0 and (re.match("\d+\.$", before_space) or re.match("[A-Z]\.$", before_space)
        or any([before_space.startswith("%s."%roman) for roman in ["II", "III", "IV", "VI", "VII", "VIII", "IX"]])):
            continue
        post_ents.append(Span(doc, ent.start, ent.end, ent.label))
    doc.ents = post_ents
    return doc
