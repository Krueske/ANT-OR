# Complex Text Classification Annotation Rules

## I. Annotation Requirements

The dataset to be annotated includes: **Knowledge Type Dataset**.

- **Knowledge Type Dataset**: Annotate which knowledge type each corpus content belongs to, such as "Policies & Regulations / Industry Standards / Life Encyclopedia / Service Guides / etc." There are a total of 15 knowledge type categories.

Please identify the **1 most appropriate data type** for each corpus content.

### Data Types:
1. Policies & Regulations
2. Industry Standards
3. Life Common Sense
4. Encyclopedia Knowledge
5. Statistical Data
6. Time-sensitive News
7. Standard Procedures
8. Scenario-based Guides
9. Operation Tutorials
10. Strategy Tutorials
11. Q&A Dialogues
12. Reviews & Ratings
13. Case Reports
14. Insights & Opinions
15. Solutions

---

## II. Category Definitions

### Factual Rules

#### 1. Policies & Regulations

**Core Definition**

Formal regulatory documents with mandatory binding force issued by national and local governments and industry regulatory agencies. These include specific clauses, applicable scope, and implementation standards, serving as the legal basis and code of conduct for life service sectors.

**Coverage Scope**
- Laws passed by people's congresses at various levels, administrative regulations of the State Council, and ministerial rules
- Local regulations and local government rules
- Mandatory standards issued by industry regulatory agencies
- Implementation details and supplementary regulations of policies (e.g., implementation rules for "Beijing Municipal Household Waste Management Regulations")
- Currently effective policy documents and historical versions (temporal status must be annotated)

**Boundary Notes**
- **Include**: Official red-headed documents with document numbers, content published in government gazettes, original policy texts published on official websites
- **Exclude**: Policy interpretations (classified under Situational Experience Knowledge - Insights & Opinions), policy change notifications (classified as Time-sensitive News), simulated policy documents from non-official agencies, expired or abolished regulations (require special annotation)
- **Note**: Local temporary control measures (e.g., epidemic prevention and control announcements) should be included if they have mandatory force; otherwise, classify as Time-sensitive News

**Corpus Example**

> Article 32 of the "Urban Public Transport Regulations" (State Council Order No. 709): Urban public transport enterprises shall set clear, standardized, and conspicuous signs and identifications in accordance with national unified standards to provide necessary information services for passengers.

---

#### 2. Industry Standards

**Core Definition**

Non-mandatory technical standards and service guidelines formulated by industry associations and standardization organizations to regulate enterprise behavior within the industry, ensure service quality, and promote healthy industry development.

**Coverage Scope**
- National standards (GB), industry standards (e.g., QB light industry standards)
- Association standards (standards beginning with T/)
- Enterprise alliance standards
- Service specification manuals (e.g., "Domestic Service Standards")
- Professional technical operation guidelines (e.g., "Food Service Food Safety Operation Standards")

**Boundary Notes**
- **Include**: Officially published standard texts, service processes certified by industry associations
- **Exclude**: Internal regulations formulated by enterprises themselves, unapproved "folk standards"
- **Note**: Distinguish from Policies & Regulations: No legal mandatory force

**Corpus Example**

> Article 5.2 of T/CHIA 002-2020 "Internet + Medical Service Evaluation Standards": Online consultation platforms shall establish doctor qualification verification mechanisms to ensure the authenticity and validity of practicing physician registration information, with verification frequency no less than once per quarter.

> Supporting standards for "Hotel Industry Public Security Management Measures": Three-star and above hotels shall be equipped with facial recognition systems, with check-in registration data connected to public security systems in real-time.

---

#### 3. Life Common Sense

**Core Definition**

Basic knowledge of daily life that is universally recognized and accumulated over time by society, covering health, safety, consumption, and other fields, with universality and foundational characteristics.

**Coverage Scope**
- Personal health maintenance knowledge (e.g., principles of balanced diet)
- Family safety common sense (e.g., handling gas leaks)
- Basic consumer rights knowledge
- Social etiquette norms
- Seasonal life tips (e.g., heatstroke prevention knowledge)

**Boundary Notes**
- **Include**: General knowledge verified through authoritative channels, life tips cross-validated from multiple sources
- **Exclude**: Professional medical advice (classified under Encyclopedia Knowledge), regional customs (classified under Situational Experience Knowledge - Case Reports)
- **Note**: Need to distinguish between "common sense" and "personal experience." For example, "leftover food shouldn't be eaten" should only be included if supported by scientific evidence.

**Corpus Example**

> Summer air conditioning temperature setting recommendation: Indoor temperature should be maintained at 26°C or above, with the temperature difference between indoor and outdoor not exceeding 7°C, to avoid triggering "air conditioning sickness."

> Food refrigeration storage time: Cooked food should not be stored for more than 3 days in a 4°C refrigerator; fresh meat should not be frozen for more than 3 months.

---

#### 4. Encyclopedia Knowledge

**Core Definition**

Systematic and professional objective knowledge collections sourced from authoritative encyclopedias, academic works, or professional institutions, with complete knowledge systems and reliable sources.

**Coverage Scope**
- Medical health knowledge (e.g., disease symptoms, treatment principles)
- Geographic and environmental knowledge (e.g., characteristics of urban areas)
- Professional term explanations (e.g., meaning of "commercial registration")
- Institutional background information (e.g., functions of the "12315 platform")
- Product ingredient explanations (e.g., food nutrition label interpretation)

**Boundary Notes**
- **Include**: Encyclopedia entries, government white paper data, knowledge documents published by professional institutions
- **Exclude**: Personal experience summaries (classified under Situational Experience Knowledge), non-professional "popular science" content (requires verification)
- **Note**: Distinguish from Industry Standards: Not binding; Distinguish from Statistical Data: Not quantitative information

**Corpus Example**

> Composition of social insurance "Five Insurances and One Fund": Pension insurance (employer 16% + individual 8%), Medical insurance (employer 9.8% + individual 2%), Unemployment insurance (employer 0.7% + individual 0.3%), Work-related injury insurance (employer 0.4%-1.9%), Maternity insurance (employer 0.8%), Housing provident fund (employer 5%-12% + individual 5%-12%).

> Norovirus transmission routes: Mainly transmitted through fecal-oral route, can be transmitted through contaminated food, water, object surfaces, and aerosols, with an incubation period of 12-48 hours.

---

#### 5. Statistical Data

**Core Definition**

Quantitative information obtained through standardized statistical methods, including clear time ranges, statistical scopes, and data sources, reflecting social and economic operation conditions.

**Coverage Scope**
- Government statistical bulletins (e.g., GDP, CPI data)
- Industry operation data (e.g., tourism visits, catering revenue)
- Livelihood service indicators (e.g., bus punctuality rate, registration waiting time)
- Special survey data (e.g., consumer satisfaction reports)
- Historical data comparisons (e.g., housing price trends over the past five years)

**Boundary Notes**
- **Include**: Raw data with clear statistical methodology descriptions, data reports published by authoritative institutions
- **Exclude**: Estimated data (e.g., "approximately 80% of users believe..." without survey basis), online voting results
- **Note**: Must have clear data sources

**Corpus Example**

> National Bureau of Statistics 2023 data: National per capita disposable income was 39,218 yuan, a nominal increase of 6.3% year-on-year; urban per capita consumption expenditure was 29,663 yuan, a year-on-year increase of 9.4%.

> Ministry of Transport monitoring: On the first day of the 2024 May Day holiday, the total number of passengers sent by national railways, highways, waterways, and civil aviation was 57.857 million, a 13.5% increase compared to the same period in 2023.

---

#### 6. Time-sensitive News

**Core Definition**

News information reflecting real-time dynamics of social life, published by media or information platforms, emphasizing timeliness and event-based characteristics, without long-term reference value.

**Coverage Scope**
- Life service news reported by mass media
- Commercial platform promotional activity notifications
- Emergency event progress (e.g., latest updates on mall fires)
- Seasonal service adjustments (e.g., summer opening hours for scenic spots)
- Market price changes (e.g., oil price adjustment notifications)

**Boundary Notes**
- **Include**: Fact-based reports published by mainstream media, verified platform announcements
- **Exclude**: Original policy texts (classified under Policies & Regulations), in-depth analysis (classified under Situational Experience Knowledge - Insights & Opinions)

**Corpus Example**

> Beijing News, April 28, 2024: During the May Day holiday, multiple Beijing subway lines will extend operations. From April 30 to May 5, 8 lines including Line 1 and Line 2 will extend operating hours until after 24:00.

> Meituan Waimai announcement: From now until May 5, Beijing area delivery fees will implement a dynamic adjustment mechanism, with base delivery fees increasing by 15%-30% during severe weather.

---

### Procedural Process Categories

#### 7. Standard Procedures

**Core Definition**

Officially regulated standardized procedures for handling affairs, with certainty, mandatory nature, and universal applicability, suitable for service acquisition paths in most standard scenarios.

**Coverage Scope**
- Government service guides (e.g., ID card processing)
- Public service application procedures (e.g., public rental housing application)
- Statutory approval processes (e.g., food business license)
- Standard service operation steps (e.g., bank account opening)
- Cross-department collaborative processes (e.g., cross-region medical insurance filing)

**Boundary Notes**
- **Include**: Government-published procedures, industry standard operating procedures, statutory required steps
- **Exclude**: Personalized alternative solutions (classified under Situational Experience Knowledge - Solutions), non-essential step recommendations

**Corpus Example**

> Beijing Residence Permit standard procedure:
> 1. Log in to the "Beijing Tong" APP to submit application → 2. Schedule on-site verification time → 3. Bring original ID card and residence proof to the street office for verification → 4. Electronic residence permit issued within 15 working days.

> Enterprise social security account opening procedure:
> 1. Complete registration through the "National Enterprise Credit Information Publicity System" → 2. Log in to the social security online service platform to submit account opening application → 3. Receive social security registration certificate within 10 working days → 4. Declare and pay monthly.

---

#### 8. Scenario-based Guides

**Core Definition**

Customized service acquisition paths for specific conditions or special scenarios, including conditional judgments and branch processing, solving personalized needs that standard procedures cannot cover.

**Coverage Scope**
- Special population service channels (e.g., elderly green channels)
- Abnormal situation handling processes (e.g., remedies for missing documents)
- Time and space limited scenario solutions (e.g., holiday special services)
- Multi-condition combination paths (e.g., "non-local household registration + rental" settlement guide)
- Emergency alternative solutions (e.g., offline processing during system failures)

**Boundary Notes**
- **Include**: Officially published special situation handling guidelines, verified alternative paths
- **Exclude**: Personal temporary solutions (classified under Situational Experience Knowledge), unverified "tips"

**Corpus Example**

> Cross-region emergency medical insurance settlement scenario-based guide (trigger condition: insured person with sudden illness in another location):
> 1. First call the social security bureau of the insured location for record-filing → 2. Keep all original medical receipts → 3. Upload materials through the "National Medical Insurance Service Platform" APP within 30 days after discharge → 4. After review, expenses will be directly settled to the bank card.

> Hospital visit guide during red rainstorm warnings:
> ① Non-emergency patients recommended to postpone visits → ② Emergency patients prioritize hospitals with subway connections (check real-time flooding maps) → ③ Patients driving use the B2 underground parking entrance (avoid surface backflow risks).

---

#### 9. Operation Tutorials

**Core Definition**

Usage guidance for specific platforms or tools, including interface operation steps, function descriptions, and common problem handling, with practical operability and technical details.

**Coverage Scope**
- Government service platform operation guides (e.g., "Jing Tong" mini-program usage)
- Commercial APP function tutorials (e.g., Meituan medicine ordering)
- Smart device usage instructions (e.g., self-service registration machine operation)
- Online processing tips (e.g., electronic license uploading)
- System troubleshooting guides (e.g., facial recognition failure handling)

**Boundary Notes**
- **Include**: Requires specific operational actions
- **Exclude**: Folk experience (classified under Situational Experience Knowledge - Strategy Tutorials), feature improvement suggestions

**Corpus Example**

> "Beijing Health Kit" nucleic acid appointment operation tutorial (April 2024 version):
> 1. Open WeChat and search for "Beijing Health Kit" mini-program → 2. Click "Nucleic Acid Appointment" to enter the service page → 3. Select testing institution and time slot (note: some locations in Chaoyang District require 1-day advance appointment) → 4. Fill in personal information and submit → 5. Screenshot and save the appointment QR code (valid for 24 hours).

> Alipay "Citizen Center" social security query operation:
> Home page → Search "Citizen Center" → Select your city → Click "Social Security" → Enter ID number and verification code → Can query payment records and insurance certificates (requires bank card verification).

---

### Situational Experience Knowledge Categories

#### 10. Strategy Tutorials

**Core Definition**

Service acquisition techniques and optimization paths summarized by users based on personal practice, including personalized strategies and practical details, emphasizing practicality and efficiency improvement.

**Coverage Scope**
- Service experience optimization solutions (e.g., hospital fast registration tips)
- Time/cost saving methods (e.g., off-peak processing guides)
- Hidden feature挖掘 (e.g., government APP shortcut operations)
- Resource integration strategies (e.g., multi-platform price comparison tips)
- Problem prevention measures (e.g., document preparation checklist templates)

**Boundary Notes**
- **Include**: User experience verified by multiple people, sharing with specific operational details
- **Exclude**: Official operation guides (classified under Procedural Process - Operation Tutorials), theoretical suggestions without practice

**Corpus Example**

> 【2024 Beijing Obstetrics and Gynecology Hospital Registration Strategy】Tested and effective:
> - Best registration time: 8 weeks + 3 days of pregnancy (avoid Friday/Sunday system updates)
> - Online appointment grabbing tips: Fill in basic information in advance, refresh 5 seconds before countdown, selecting "Obstetrics General Number" has higher success rate
> - Document preparation: In addition to regular materials, bring 1 extra 1-inch photo (for on-site form filling), can save 15 minutes
> - Hidden channel: When unable to get an appointment, go directly to the 3rd floor obstetrics clinic to find the head nurse, who has 5 daily on-site additional appointment slots.

> 【Government Hall Efficient Service Strategy】:
> 1. Make an appointment 1 day in advance on the "Beijing Tong" APP, select the "9:00-9:30 AM" time slot (fewest people) → 2. Go directly to window 3 and say "I have an appointment with Wang Ming," can skip queuing for a number → 3. Bring 3 sets of document copies to avoid reprinting on-site → 4. Request an "Acceptance Receipt" after processing, can check progress on WeChat.

---

#### 11. Q&A Dialogues

**Core Definition**

Records of communication between real users and service providers (or community members), focusing on specific problem solving, including problem context and targeted responses.

**Coverage Scope**
- Government service hotline conversations (e.g., 12345 consultation records)
- Commercial platform customer service chat records
- Community Q&A platform exchanges (e.g., Zhihu, Xiaohongshu Q&A)
- User mutual help group chat fragments
- Expert online consultation records
- Other constructed dialogue datasets

**Boundary Notes**
- **Include**: Two-way communication, including problem context and solutions
- **Exclude**: One-way information releases, pleasantries without substantive content

**Corpus Example**

> **User**: I'm applying for public rental housing on the "Beijing Tong" APP, and it prompts "household registration information abnormal," but my household registration book is normal. What should I do?
> 
> **Customer Service**: Hello, please check if the registered address is consistent with the system registration. Recently, the Dongcheng District household registration system was upgraded, and some users need to manually update to the latest address format (example: original "No. 100 Dongsi North Street" needs to be changed to "No. 100A Dongsi North Street"), which you can modify in "Personal Center - Household Registration Information" and resubmit.

> **User**: When ordering medicine on Meituan, why does it show "nearby pharmacy out of stock"? I clearly saw the pharmacy across the street was open?
> 
> **Meituan Customer Service**: There may be three reasons: 1. The pharmacy has not opened Meituan delivery service (can call to confirm); 2. The medicine is a prescription drug and you have not uploaded a prescription; 3. The delivery person is currently picking up in the store, and the system display is delayed. Suggestions: First check the pharmacy homepage "Service Scope," or directly call the pharmacy to reserve the medicine.

---

#### 12. Reviews & Ratings

**Core Definition**

Subjective feedback from users on service processes or results, including satisfaction evaluations, advantages and disadvantages analysis, and improvement suggestions, reflecting real experience feelings.

**Coverage Scope**
- Commercial service platform ratings (e.g., Dianping hospital reviews)
- Government service satisfaction feedback (e.g., "Good/Bad Review" system data)
- Social media service experience sharing (e.g., Weibo complaints)
- Service comparison evaluations (e.g., different bank APP experience comparisons)
- Problem complaints and resolution feedback

**Boundary Notes**
- **Include**: Evaluations with specific service scenarios, feedback containing improvement suggestions
- **Exclude**: Advertising content, subjective judgments without service basis

**Corpus Example**

> 【Tertiary Hospital Internet Hospital Evaluation】3.0 (April 15, 2024)
> - Pros: Doctors respond quickly (within 1 hour), electronic prescriptions directly pushed to pharmacies;
> - Cons: Follow-up visits must select the original doctor, system does not support automatic matching; refund process is complex, requires manual phone operation;
> - Suggestions: Add "similar doctor recommendation" function, optimize refund automation process.

> 【Government APP Experience】5.0 (April 22, 2024)
> Used the "Jing Tong" mini-program to handle residence permit renewal, completed in 10 minutes! Much better than going to the street office and queuing for 2 hours last year. Special praise for the "Smart Document Pre-review" function, which automatically fills information after uploading ID card, only needed to supplement 1 item. But suggest adding active push for processing progress, don't always make me check myself.

---

#### 13. Case Reports

**Core Definition**

Complete process records and analysis of typical service scenarios, including background, process, results, and insights, with reference value and replicability.

**Coverage Scope**
- Service dispute resolution cases (e.g., consumer rights protection processes)
- Complex matter processing records (e.g., cross-province medical insurance transfer)
- Special needs service records (e.g., experiences of people with disabilities handling affairs)
- Success/failure experience summaries (e.g., rapid settlement cases)
- Industry typical service scenario reviews

**Boundary Notes**
- **Include**: Must have time/location/person elements
- **Exclude**: Fragmented experience sharing, unverified "legends"
- **Note**: Need to annotate case timeliness and regional applicability to avoid generalization

**Corpus Example**

> 【Case Report】2023 Beijing Cross-region Medical Insurance Record-filing Failure Processing Record
> - **Background**: Shandong insured person in Beijing Cancer Hospital emergency room, without prior record-filing
> - **Process**: ① Explain situation at emergency payment, hospital marks "not filed"; ② Next day call Shandong Medical Insurance Bureau at 0531-12393 for supplementary filing; ③ Provide emergency diagnosis certificate and expense list; ④ Upload materials through "National Medical Insurance Service Platform" APP
> - **Result**: 65% reimbursement after 15 working days, reducing loss by 21,000 yuan compared to self-payment
> - **Key Point**: Must complete supplementary filing within 48 hours, emergency materials must include diagnosis with "emergency" wording.

> 【Failure Case】Community Elderly Care Station Application Rejection Analysis
> - **Applicant Conditions**: 80-year-old living alone, with Beijing household registration
> - **Problem Link**: Did not provide "Home Safety Assessment Report for the past 6 months" (new regulation implemented October 2023)
> - **Remedy**: Contacted street office to commission third-party agency for on-site assessment (cost 200 yuan), resubmitted after 3 days
> - **Lesson**: After policy updates, the community did not promptly notify elderly residents, suggesting the establishment of an active reminder mechanism.

---

#### 14. Insights & Opinions

**Core Definition**

In-depth analysis and insights into service phenomena, policy impacts, or industry trends, reflecting thinking dimensions and value judgments.

**Coverage Scope**
- Policy impact interpretations (e.g., in-depth analysis of medical insurance reform)
- Industry trend judgments (e.g., development direction of smart government services)
- Service model critiques (e.g., excessive formalization of "fingertip government services")
- User behavior research (e.g., reports on the digital divide for elderly people)
- Cross-domain correlation analysis (e.g., transportation policy and medical accessibility)

**Boundary Notes**
- **Include**: In-depth analysis based on facts, logically supported opinion statements
- **Exclude**: Simple emotional venting, unverified speculation
- **Note**: Distinguish between objective analysis and subjective assumptions, not purely factual statements (distinguished from authoritative releases)

**Corpus Example**

> 【Policy Interpretation】Deep Impact of 2024 Medical Insurance Catalog Adjustment:
> This addition of 111 types of medicines, but innovative drugs still face the "last mile" problem in access. The hospital procurement link's "drug proportion" assessment was not simultaneously adjusted, resulting in an actual implementation rate of less than 30% for high-priced drugs like PD-1 inhibitors. Suggestion: Establish a "medical insurance negotiation - hospital procurement" linkage mechanism, incorporating innovative drug use into public hospital performance assessments.

> 【Industry Insight】Hidden Thresholds Behind "One-Stop Government Services":
> On the surface, government digitalization improves efficiency, but surveys show that online service success rates for people over 65 are only 28%. The real digital divide is not in device access, but in service design: facial recognition liveness detection is not friendly to cataract patients; "electronic licenses" promotion ignores the reality of low printer penetration in rural areas. The solution should be "digital + manual" dual-track services, rather than simply pursuing online processing rates.

---

#### 15. Solutions

**Core Definition**

Systematic response strategies proposed for specific service problems, including problem diagnosis, solution steps, and effect verification, emphasizing operability and results orientation.

**Coverage Scope**
- Service bottleneck breakthrough solutions (e.g., comprehensive management of hospital registration difficulties)
- Process optimization designs (e.g., government service "one-window acceptance" transformation)
- Technology empowerment solutions (e.g., AI smart form-filling tool development)
- Personalized problem responses (e.g., handling special missing materials)
- Cross-department collaboration mechanisms (e.g., medical insurance - hospital - pharmacy data connectivity)

**Boundary Notes**
- **Include**: Verified effective methods, replicable solution frameworks
- **Exclude**: Vague suggestions, theoretical concepts without practice

**Corpus Example**

> "Emergency Plan for Sudden Nanny Departure: ① Contact agency → ② Temporary hourly worker → ③ Contract accountability"

> "Dealing with Upstairs Water Leakage: 1. Collect evidence 2. Negotiate 3. Litigate 4. Enforce"

---

## III. Boundary Delineation

### 1. Distinctions Between Standardized Steps, Operation Procedures, and Strategy Experience

**a. Standardized Steps**

Official mandatory behavioral norms with legal effect.

> Example: "Residence permit application must follow this order: 1. Swipe ID card to get number → 2. Submit original and 1 copy (both required) → 3. Collect within 5 working days"

**b. Operation Procedures**

Coherent action guides provided by officials/platforms for completing tasks.

> Example: "Beijing Health Kit nucleic acid appointment: 1. Log in to APP → 2. Select testing point (Chaoyang District requires 1-day advance) → 3. Screenshot and save QR code (valid for 24 hours)"

**c. Strategy Experience**

Efficiency optimization tips summarized from user testing.

> Example: "Efficient government hall service: ① Avoid Monday peaks → ② Say 'I have an appointment with Wang Ming' to skip queuing → ③ Bring 3 copies to avoid reprinting"