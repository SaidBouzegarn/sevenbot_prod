import os
import asyncio
from typing import List
from pydantic import BaseModel
from openai import AsyncOpenAI

class ArticleInsights(BaseModel):
    insights: List[Insight]

async def extract_article_insights(article: dict[str, str]) -> dict[str, any]:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""
    Analyze the following article and extract key insights. Each insight should be a complete, self-contained sentence that captures a significant piece of information from the article. The insights should collectively cover all the important information in the article, even if this means some information is repeated across multiple insights.

    Article:
    {article['article_body']}

    Instructions:
    1. Read and analyze the entire article carefully.
    2. Extract key information, facts, arguments, and conclusions.
    3. Formulate each insight as a complete, standalone 3-sentence mamximum paragraph.
    4. Ensure that all important information from the article is captured in the insights.
    5. Include specific details, numbers, and quotes when relevant.
    6. Aim for clarity and precision in each insight.
    7. Do not include your own opinions or external information not present in the article.
        Output the insights as a list of strings, with each insight being a separate item in the list.

    Example : 
    
    article_text = "Millions of lives have been saved, millions of people are on treatment, and millions of babies have been born HIV-free. We have achieved a lot, but there is still work to do.
    Since its establishment in 2003, the US President’s Emergency Plan for AIDS Relief (PEPFAR) has saved over 25 million lives and drastically reduced HIV infection rates through antiretroviral treatments. Now well established as the leading global funder of HIV prevention efforts, PEPFAR is supporting high HIV burden countries to meet the UNAIDS 95-95-95 targets by 2025 and aiming to end HIV as a public health threat by 2030.
    In conversation with PharmaBoardroom, PEPFAR's Dr. Emily Kainne Dokubo emphasizes the significance of partnerships in achieving PEPFAR's mission - extending PEPFAR's reach and improving health systems globally - and how it is working to address paediatric HIV disparities, given children still remain disproportionately affected by HIV/AIDS.
    Could you give us a brief overview of your career trajectory and the scope of your role at PEPFAR?
    I am a physician and an epidemiologist by training and previously served as the US Centers for Disease Control and Prevention (CDC) director in Cameroon and in the Caribbean region. Throughout my career, I have had a focus on global health, primarily around addressing the HIV and tuberculosis (TB) pandemics through both clinical practice and research. I currently serve under Ambassador Dr. John Nkengasong as the Deputy US Global AIDS Coordinator for Program Quality for the US President's Emergency Plan f...
    What have been PEPFAR's biggest impacts on the battle against HIV in the two decades since its foundation?
    At the start of the HIV pandemic over 40 years ago there were rising new infections, ongoing transmission, and high morbidity and mortality, especially on the African continent due to limited access to prevention and treatment options. In response, during his 2003 State of the Union Address, then-US President George W. Bush announced the establishment of PEPFAR, saying, "seldom has history offered a greater opportunity to do so much for so many." PEPFAR has lived up to that commitment. In the 21 years si...
    What will PEPFAR's priorities be over the next five and a half years as the entire HIV community works towards the very ambitious UNAIDS targets of ending HIV as a public health threat by 2030?
    PEPFAR has already helped to change the trajectory of the HIV pandemic over the past two decades. The data bears out that all-cause mortality has been lowered by a greater margin in PEPFAR-supported countries than in non-PEPFAR supported countries. In the short term – by 2025 – we want to support as many high HIV burden countries as possible achieve the 95-95-95 targets. This is beneficial not just for an individual's own health but is also a public health measure for reducing HIV transmission. HIV in ad...
    How does PEPFAR choose the countries on which it focuses its efforts?
    One hallmark of PEPFAR is its data-driven approach. We work with host governments to support their response efforts in places with the highest disease burden. At the start of PEPFAR, this was predominantly in sub-Saharan Africa, where we still support a lot of countries, but we also work in Asia, the Caribbean, and South and Central America. PEPFAR, as part of the Bureau of Global Health Security and Diplomacy within the US Department of State, has a truly global reach and extends beyond HIV to support r...
    How does the fact that PEPFAR's funding has recently only been secured for one year instead of the usual five affect its operations and long-term planning?
    Since its establishment, PEPFAR has always counted on bipartisan support. Eleven congresses and four presidents have given their support to this lifesaving program, which we are pleased to see continue with PEPFAR's recent reauthorisation. We look forward to working with Congress on a clean, five-year reauthorisation which would demonstrate the US government's longstanding commitment to end HIV/AIDS as a public health threat by 2030. This would allow us to plan for the longer term and continue to provide...
    How is PEPFAR adjusting its approach to ensure that more people can benefit from the HIV prevention and treatment solutions it helps provide?
    One of the benefits of PEPFAR is that we have a clear mandate. We remain focused on the program's founding mission and are still just as committed to serving all communities that we work with without discrimination. This is especially true for those key and priority populations that are at increased risk of HIV acquisition. Reaching those underserved key populations is critical for advancing our global HIV/AIDS response. To do so, in potentially challenging environments, we are strengthening our support ...
    PEPFAR is the major funder of HIV prevention efforts, but how important are partnerships with other stakeholders in carrying out its mission?
    Extremely. This is a huge effort that cannot be achieved by any single entity. The partnerships that we have formed are one of our key strengths. This includes partnerships with other departments and agencies within the US government; multilateral organisations such as WHO, UNAIDS, the Global Fund to Fight AIDS, Tuberculosis and Malaria, and UNICEF; and private entities. One success story from our public-private partnerships is the development and introduction of new long-acting options to prevent HIV in...
    The pharmaceutical industry tends to launch new treatments in developed Western markets at a high price to recoup their R&D investments before they are later brought to places like Africa at a lower price or as a generic. HIV necessitates a different approach, so how do you go about engaging pharma to rapidly bring new products to the patients that need them most?
    The beauty of PEPFAR's partnerships is that we have been able to engage in good faith. We have been fortunate that many of our partners understand the importance of our mission. In many of the places where we work, governments have limited ability to provide treatment to their populations on their own. Therefore, PEPFAR, by purchasing HIV-related commodities, brings in the volume and guarantees our pharmaceutical partners that we will procure to supply to those countries. For example, PEPFAR is rolling o...
    Even without a vaccine, we already have the prevention and treatment tools at our disposal to achieve the 2030 goals, However, the difficulty lies in access and delivery. What do you see as the key gaps to getting these treatments to an even wider swathe of the population that needs them?
    Having the right systems in place is crucial to the sustainability of HIV programs and ensuring the quality of HIV service provision. Commodities do not deliver themselves. That is why PEPFAR is also focused on health system strengthening: PEPFAR support has helped train over 346,000 health workers and we also help bolster sample delivery systems, drug delivery systems, delivery to remote communities, and third-party logistics. Reaching clients where they are is also key. Every individual has a personal...
    "

    # List of insights extracted from the article
    insights = [
        "Since 2003, PEPFAR has saved over 25 million lives and significantly reduced HIV infection rates through antiretroviral treatments. It is the leading global funder for HIV prevention and aims to end HIV as a public health threat by 2030.",
        "Dr. Emily Kainne Dokubo emphasizes the role of partnerships in PEPFAR's mission, particularly in addressing disparities in pediatric HIV treatment, as children remain disproportionately affected by HIV/AIDS.",
        "PEPFAR has supported the development of health systems in over 55 countries, helping nations achieve the UNAIDS 95-95-95 targets for HIV testing, treatment, and viral suppression by 2025.",
        "In the 21 years since its establishment, PEPFAR has provided antiretroviral treatments to nearly 21 million people and facilitated the birth of 5.5 million babies free of HIV.",
        "The short-term goal of PEPFAR is to help high HIV-burden countries achieve the UNAIDS 95-95-95 targets by 2025, which will improve individual health and reduce HIV transmission globally.",
        "PEPFAR uses a data-driven approach to prioritize high-burden countries for intervention, predominantly in sub-Saharan Africa, but also extending to Asia, the Caribbean, and Central and South America.",
        "PEPFAR’s funding has typically enjoyed bipartisan support, but recent reauthorization for only one year complicates long-term planning. A five-year reauthorization would provide stability for HIV prevention and treatment efforts.",
        "PEPFAR remains committed to serving key populations at increased risk of HIV infection, particularly through a person-centered, differentiated care model that adjusts services to meet individual needs.",
        "Partnerships are essential to PEPFAR’s work, including collaborations with other US government agencies, international organizations like WHO and UNAIDS, and private sector entities to develop new HIV prevention tools such as long-acting PrEP.",
        "PEPFAR helps pharmaceutical companies rapidly introduce HIV-related commodities to countries with limited healthcare resources by guaranteeing bulk purchases, facilitating access to new prevention and treatment tools.",
        "A lack of infrastructure and healthcare delivery systems continues to hinder broader access to HIV prevention and treatment, which is why PEPFAR is also focused on health system strengthening, including the training of 346,000 healthcare workers.",
        "Despite successes in adult populations, pediatric HIV remains a challenge, with many children and adolescents lacking access to treatment. PEPFAR has launched initiatives to improve early infant diagnosis and treatment for HIV-positive mothers.",
        "PEPFAR's $20 million Youth Initiative aims to increase HIV awareness and prevention among adolescents, empowering the younger generation to manage their health and prevent new HIV infections.",
        "Although global efforts have transformed HIV from a death sentence to a manageable disease, 1.5 million new infections occur annually, and 600,000 people die from HIV-related complications. Continued efforts are essential to ending HIV as a public health threat by 2030."
    ]

    """

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a highly skilled AI assistant specializing in extracting insights from articles."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max(1000, len(article['article_body']) * 4),
        temperature=0.315,
        response_model=ArticleInsights
    )

    insights = completion.choices[0].message.content

    return {
        "title": article.get('title', 'N/A'),
        "insights": [insight.content for insight in insights.insights]
    }
