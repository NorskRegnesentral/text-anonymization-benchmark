# Annotation Guidelines
------

This text provides annotation guidelines for the anonymisation of human rights court rulings with the Tagtog tool.  _Please read the instructions below carefully._

You are given a collection of court cases (in a directory with your name), each associated with the name of a specific individual to anonymise from the case. The goal of your annotation is to mark all text spans (= a continuous stretch of one or more words) that correspond to an entity of a certain category and, afterwards indicate whether they can be used to re-identify the individual to be protected.

The data has been *pre-annotated* automatically. Your annotation work consists of four stages:

- **Step 0:** Read through the entire case once
- **Step 1: Entities.** Taking the pre-annotations as a starting point, annotate each entity with a semantic type (PERSON, CODE, LOC, ORG, DEM, DATETIME, QUANTITY, MISC, see below). You need to remove/correct errors in the pre-annotated entities, and manually annotate entities that went undetected. 
- **Step 2: Masking.** For each entity annotated in Step 1, specify whether their identifier type (direct identifier, quasi-identifier, or does not need masking) and their confidential status. See below for detailed definitions of these terms.
- **Step 3: Final review.** Save and confirm your annotation, and inspect the review document generated in your `masked` subfolder to ensure you have not missed any identifier or confidential attribute. Otherwise, go back to Step 1 or 2. 

**Starting from Monday, March 29**, we will do a quality check of the annotations in pairs. See at the bottom of this document for details. 

## Step 1: Entities

Step 1 focuses on annotating entities with semantic types. To facilitate your task, we provide pre-annotations generated automatically.  *Those pre-annotations are just a starting point, and are far from perfect: you <u>must</u> actively seek to correct and extend the annotation*</u>. This means that you should always:
1. Verify whether each pre-annotated entity is correct or needs to be removed or edited (either because it is the wrong type, or whether the span should be changed).
2. Manually annotate each entity that was not detected by the automatic tool. For some categories such as DEM or MISC, the automatic tool will only detect a small fraction of the entities, so you should expect to find some in every document.
3. If two mentions refer to the same underlying entity but have a different string (e.g. "John Smith" and "Mr. Smith"), insert a reference *relation* between them (see below).

For Step 1, you do not need to worry about *re-identification risks* (this will come in Step 2), you just need to mark all entities pertaining to one of the provided categories.

Entities that occur several times through a document are visually marked in TagTog with a yellow border for all mentions except the first one (to indicate that the subsequent mentions are "derived" from the first one). When an entity needs to be corrected or removed, you should therefore apply your changes on the first mention - this way, your changes will be automatically propagated to all mentions, without having to repeat your corrections one by one.

The list of categories is presented below. If the information does not fit within one of these, the MISC category should be used.

<table border="1">
  <tr>
    <th bgcolor="#dddddd">Category</th>
    <th bgcolor="#dddddd">Description</th>
  </tr>
  <tr>
    <td>PERSON</td>
    <td>Names of people, including nicknames/aliases, usernames and initials</td>
  </tr>
  <tr>
    <td>CODE</td>
    <td>Numbers and codes that identify something, such as SSN, phone number, passport number, license plate</td>
  </tr>
  <tr>
    <td>LOC</td>
    <td>Places and locations, such as: 
<ul>
  <li>Cities, areas, countries, etc.</li>
  <li>Addresses</li>
  <li>Named infrastructures (bus stops, bridges, etc.)</li>
</ul>
</td>
  </tr>
  <tr>
    <td>ORG</td>
    <td>Names of organisations, such as:
<ul>
  <li>public and private companies</li>
  <li>schools, universities, public institutions, prisons, healthcare institutions
non-governmental organisations, churches, etc.
</li>
</ul>
</td>
  </tr>
<tr>
    <td>DEM</td>
    <td>Demographic attribute of a person, such as:
<ul>
  <li>Native language, descent, heritage, ethnicity</li>
  <li>Job titles, ranks, education</li>
<li>Physical descriptions, diagnosis, birthmarks, ages</li>
</ul>
</td>
  </tr>
  <tr>
    <td>DATETIME</td>
    <td>Description of a specific date (e.g. October 3, 2018), time (e.g. 9:48 AM) or duration (e.g. 18 years). 
</td>
  </tr>
  <tr>
    <td>QUANTITY</td>
    <td>Description of a meaninful quantity, e.g. percentages or monetary values.
</td>
  </tr>
  <tr>
    <td>MISC</td>
    <td>Every other type of information that describes an individual and that does not belong to the categories above</td>
  </tr>
  </tr>
  <tr>
</table>

### Examples

In general, annotation should mark the *minimal span*  which denotes the entity or property in question. 

#### PERSON
For person names the annotation should include titles and honorifics, such as Mr., Dr., etc. since these may contribute to the precise identification of an individual.
* <u>Mr Gestur Jónsson</u> and <u>Mr Ragnar Halldór Hall </u>
* <u>Mr and Mrs Smith</u>

Some examples which are regarded as names:
* Names: E.g., ‘Harry Hole’, 'Hole', 'Harry'
* Initials: E.g. 'H.H.'
* Spelling mistakes: E.g. 'Hary Hole'
* All orthographic variations E.g. 'harry hole', 'Harry HOLE'

**NB:** The name of a person can be both a direct identifier (if it is the full name of the individual to protect) or a quasi-identifier (if it is the name of another related person).

#### CODE
Includes ID-numbers and codes, in particular numbers and codes that identify something, such as SSN, phone number, passport number, license plate, report identifiers etc. Codes can be either identifiers or quasi-identifiers, according to whether they unequivocally refer to the individual to protect or not.

* an application (no. <u>42552/98</u>)

#### LOC
Includes cities, areas, counties, addresses, as well as other geographical places, buildings and facilities. Other examples are airports, churches, restaurants, hotels, tourist attractions, hospitals, shops, street addresses, roads, oceans, fjords, mountains, parks.
* <u>Reykjavik</u>
* <u>Øvregaten 2a, 5003 Bergen</u>

Include numbers when they are part of the entity name:
* <u>Pilestredet 48</u><sub>(LOC)</sub>
* <u>Rema 1000</u><sub>(ORG)</sub>

Always annotate the whole name, never nested parts.
<br />
Annotate like this:
* <u>Høgskolen i Oslo og Akershus</u><sub>(ORG)</sub>


And <b>not</b> like this:

* <strike>Høgskolen i <u>Oslo</u><sub>(LOC)</sub> og <u>Akershus</u><sub>(LOC)</sub></strike>

Separate entities connected with a conjunction (e.g. ‘and’) should be annotated separately:
* <u>Pamir</u> and <u>Alay</u> valleys<sub>(LOC)</sub>

In case an entity could be either ORG or LOC (e.g. Turkey or Breitvet prison), the entity type best describing the referent should be chosen, i.e. ORG if the occurrence refers to the institution itself, and LOC if it refers to a geographic location.

#### ORG
Includes any named collection of people, businesses, institutions, organizations, universities, hospitals, churches, sports teams, unions, political parties etc.

Corporate designators like AS, Co. and Ltd. are to be included as part of the name.

Translations and acronyms are included in the span, e.g. 
* <u>KCK (Koma Civakên Kurdistan – “Kurdistan Communities Union”)</u>


Definite or indefinite articles are typically not included in the span, unless explicitly part of the entity (as for titles of books and movies, such as “The Great Gatsby”).
* The <u>Supreme Court’s Appeals Leave Committee</u>
* A ship from the <u>East India Company</u>
* <u>Istanbul public prosecutor’s office</u>
* <u>Istanbul police</u>


#### DEM
These are demographic properties and include both physical, cultural and occupational/educational properties, such as various physical descriptions, diagnosis, native language, ethnicity, job titles, age, etc.

* <u>40 years</u> old
* the applicants are <u>journalists</u>
* a group of <u>left-wing</u> extremists
* diagnosed with <u>motor neurone disease</u>
* a <u>Polish</u> and <u>naturalized-French</u> physicist

Pronouns (he, she) should not be annotated to protect gender information.


#### DATETIME
Prepositions (e.g. on, at) should not be included in the span.
* <u>Monday, October 3, 2018</u>
* at <u>9:48 AM</u>
* born in <u>1947</u>

Separate entities connected by "and" should be annotated separately:
* <u>10 March</u> and <u>12 of March</u>

Except if they partially overlap:

* <u>10 and 12 of March</u>

#### QUANTITY
Units, such as currencies, should be included in the span.
* <u> $37.5 million</u>
* <u> 375 euros</u>
* <u> 4267 SEK</u>
* <u> 1000 Kilos</u>

#### MISC
Other (quasi-)identifying words such as trademarks, products, events, etc. 
All things artificially produced are regarded products. This may include more abstract entities, such as speeches, radio shows, programming languages, contracts, laws and even ideas (if they are named).

Brands are products when they refer to a product or a line of products, but organisation when they refer to the acting or producing entity.

* Criminal investigation into the death of the applicant's son

### Exceptions

The main goal of Step 1 is to find all entities related to categories described above (without worring about whether they need to be masked or not, since this is the goal of Step 2). However, there are some entities that are technically part of those categories, but are so obviously irrelevant to the task that they can be safely ignored.

Here are some concrete exceptions you can ignore:
- profession or title of the legal professionals involved in the case (for instance "solicitor", "legal adviser", "lawyer", etc.)
- parts of generic legal references (such as the year a particular law was passed or published).

If you find other exceptions you think would be useful to add to this list, let us know!


### Relations
Finally, if some text spans are referring to the same underlying entity through different mentions (such as “John Smith” and “Mr Smith”, or “Republic of Turkey” and “Turkey”), annotate those referential relations. More precisely:
 * a. Find an occurrence of the first mention (such as “John Smith”) and click “add relation”
 * b. Find an occurrence of the second mention (such as “Mr Smith”) and click on it. You should now see a relation between the two.

You do not need to add “same_as” relations between entities with the exact same string. It is also sufficient to annotate the relation once, even though there may be several occurrences of both mentions. 

------

## Step 2: Masking

In this stage of the annotation, you should review all text spans marked in Step 1, and specify whether they need to be masked (either as a DIRECT or QUASI identifier) to protect the identity of the individual specified in the annotation task.  You should mark all direct and quasi identifiers but _not more than those_ (we still wish to retain as much textual content as possible). 

For every entity annotated during stage 1, you should set the correct value for the following two labels:
1. **identifier_type**
* DIRECT_ID (direct identifiers): text spans that _directly_ and _unequivocally_ identify the individual to protect) in the case and should therefore be masked.
* QUASI_ID (quasi-identifiers): text spans that should be masked since they may lead to the re-identification of the individual when combined with other (not masked) quasi-identifiers mentioned in the text along with public background knowledge.
* NO_MASK: entities that are neither of the above, and should therefore not need to be masked. _Most entities will belong to this category._ Start by applying NO_MASK to the <u>least</u> precise/specific quasi-identifiers in the court case first.

2. **confidential_status**
<br />
If the information is also confidential, that is, it describes religious or philosophical beliefs, political opinions, trade union membership, sexual orientation or sex life, racial or ethnic origin, health data, genetic and biometric data, specify the category (BELIEF, POLITICS, SEX, ETHNIC, HEALTH) or NOT_CONFIDENTIAL if it is not confidential.

Note that the "Confidential Status" label is only available for a subset of entity types for which we can except confidential information.

We recommend you use TagTog's [Document review](https://docs.tagtog.net/webeditor.html#document-review) mode (press `r`) to easily go through all entities one by one without having to click on anything. When deciding on the identifier type and confidential status of an entity, make sure you select the first mention - this way, your decision will be automatically propagated to all subsequent mentions of that same entity (shown with a yellow border on TagTog). 

### Direct identifiers
= text spans that contain information that directly and unequivocally identify the individual to be protected.

**Examples**: person names (including nicknames/aliases and usernames), social security numbers, passport numbers.

### Quasi-identifiers
= Information that, in isolation, does not identify the individual to be protected but can do so, in combination with other quasi-identifiers and background knowledge. These will often refer to demographical (“a 72-years old man”) or spatiotemporal attributes (“on February 6 in Sevilla”). For instance, the combination of date of birth, gender and profession will typically allow you to find out the identity of a person. 

For a re-identification to be possible, quasi-identifiers must refer to some information that can be seen as potential “publicly available knowledge” — i.e. something that we can expect that an external person may already know on the individual or may be able to find —, and the combination of quasi-identifying information should be enough to re-identify the individual with no or low ambiguity. You should judge whether it is likely that someone could, based on public knowledge, know the quasi-identifying values of the individual to be protected. There is some room for interpretation here, but the annotator should ask themselves the question: if I wanted to find out the identify of the individual in the document, should I expect to be able to connect those pieces of information with some other knowledge sources (such as news articles, social media, census data, etc.)? and, are those pieces of information enough to re-identify the individual with no or low ambiguity? *In the vast majority of cases, you don’t actually need to do any search for those knowledge sources, your intuition will suffice.*

What do we mean by "publicly available knowledge"? In practice, anything that can be found by searching on the web. With one exception: since the texts come from the ECHR, what we consider "publicly available knowledge" should *not* include the HUDOC database itself (or derived knowledge bases) -- otherwise we would need to consider virtually every word in the text as quasi-identifiers, since it would be straightforward to do a quick search on HUDOC to find the original court case and find out the identify of the individual.

As a rule of thumb, immutable personal attributes (e.g., date-of-birth) on an individual that can be known by external entities should be considered quasi-identifiers. Circumstantial attributes may be considered quasi-identifiers or not according to the chance that external entities may know such information (e.g., current place-of-living or a hospital admission date could be, but the number of times one has gone to the grocery store in a week may not). 

Usually, only very general attribute values that encompass a large number of individuals  (e.g., country-of-birth) may be ignored, since they would match a large population of individuals and would not enable a unequivocal re-identification. This also depends on the presence of other quasi-identifiers within the same document: the larger the amount and the more concrete the information they provide, the larger the chance that they may enable re-identifications. 

### Confidential attributes

= Information is such that, if disclosed, could harm or could be a source of discrimination for the individual. 

For confidential attributes, the categories of information to be protected are: 
<table border="1">
  <tr>
    <th bgcolor="#dddddd">Category</th>
    <th bgcolor="#dddddd">Description</th>
  </tr>
    <td>BELIEF</td>
    <td>Religious or philosophical beliefs</td>
  </tr>
  </tr>
  <tr>
    <td>POLITICS</td>
    <td>Political opinions, trade union membership</td>
  </tr>
  <tr>
    <td>SEX</td>
    <td>Sexual orientation or sex life</td>
  </tr>
  </tr>
  <tr>
    <td>ETHNIC</td>
    <td>Racial or ethnic origin</td>
  </tr>
  </tr>
  <tr>
    <td>HEALTH</td>
    <td>Health, genetic and biometric data. This includes sensitive health-related habits, such as substance abuse</td>
  </tr>
  <tr>
    <td>NOT_CONFIDENTIAL</td>
    <td>Not confidential information (most entities)</td>
  </tr>
</table>

## Step 3: Final Review

Save and confirm your annotation. After saving the document, a new "review" document should be now be available in your subfolder `masked`.  For instance, if your document lies in the path `annotator1/001-87407`, a new document should now appear at `annotator1/masked/001-87407_review`. (*if that's not the case, let us know*)

This document shows which entity will end up being masked in an anonymised version of the court case. Direct identifiers and quasi-identifiers are now replaced by *****. Confidential attributes are shown in clear text but marked as entity. Inspect the text to ensure you haven't forgotten any (direct or quasi) identifier or confidential attributes. Otherwise, go back to step 1 and 2 and repeat the process. 

------

## Quality check

You will perform a quality check in pairs to review each other's annotations. The procedure to follow is relatively simple: For every *confirmed* document from your teammate, do the following:
1. Read the "review" document carefully
2. Try to answer the question: Is the masking as sufficient to conceal the identity of the person? If in doubt, try to re-identify the person by googling around (but again ignoring the ECHR case itself from what you should consider publicly available knowledge)
3. Go to the "main" document, and check for mistakes/omissions in the annotations. Such mistakes could correspond to missing spans, wrong span boundaries, wrong semantic categories, wrong masking decisions, or wrong/absent confidential status. 
4. For "obvious" mistakes or omissions (for which there is no room for multiple interpretations), you can directly edit the annotations to add the corrections
5. For more substantial disagreements where you would have made a different decision than your teammate, note down your disagreement
6. Once you are done reviewing a bulk of court cases, schedule a meeting with your teammate to discuss those disagreements, and reach a final decision by consensus
7. Apply the final edits to the main document
8. Click on confirm on both the main document and the review document (as a way of marking that the review of that document is complete)
