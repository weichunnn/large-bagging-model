import os
from typing import List
import weave

import instructor
from pydantic import BaseModel
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

weave.init("together-weave")

from extract_findings import extract_both_debates

config = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

client = instructor.from_openai(config)


class StyleList(BaseModel):
    style_description: List[str]


def generate_style_prompt(n_styles, grade_responses=None) -> list[str]:
    # Load in template style prompts
    with open("app/styles.txt", "r") as file:
        styles = file.read()
    grade_responses = extract_both_debates()
    # Take in best and worst k responses from previous iterations
    if grade_responses:
        best_k = grade_responses[0]
        worst_k = grade_responses[1]
    else:
        # Default best and worst k responses
        best_k = ["strong, logical, nuanced and well-thought out argument"]
        worst_k = ["weak, illogical, shallow and poorly thought out argument"]

    style_responses = client.chat.completions.create(
        model="gpt-4",
        response_model=StyleList,
        messages=[
            {
                "role": "system",
                "content": "You are a debating coach to help generate effective, logical and thoughtful debate styles. Be descriptive and helpful. Your goal is to help create an instructive summary to write a debate speech using formats from your knowledge base to prompt an LLM agent to generate a strong, logical, nuanced and well-thought out argument.",
            },
            {
                "role": "user",
                "content": f"""
             Here are exemplary great responses to help you analyze, dissect, and replicate their styles: {best_k}. Here are bad, ineffective, worst responses that you should avoid replicating their styles: {worst_k}. 
             You can also generate the style description inspired by these prompts using a combination of a few ideas together: {styles}.

             Generate a list of {n_styles} styles description for debate speeches. Make sure to be creative and descriptive as possible. Give at least 200 words for each style description.
             """,
            },
        ],
        temperature=0.8,
        top_p=1,
    )
    print(style_responses)
    return style_responses


# Describe with detailed explanations on the approaches as well as the philosphies they follow for the different debate styles. Keep it to at least 200 words each.
# Examples of previous debate styles
# grade_responses = [
#     """
#     # Opening Statement: Tech Companies Should Bear Calibrated Legal Liability for Social Media Content

#     Esteemed judges, honored opponents, and distinguished guests,

#     Today, we stand at a critical juncture in the evolution of our digital society. The question before us - "Should technology companies be held legally liable for content posted on social media platforms?" - is not merely academic. It strikes at the heart of how we balance innovation, free expression, and social responsibility in the digital age.

#     I argue that technology companies should indeed bear legal liability for content on their platforms, but this liability must be carefully calibrated. Let me outline why this nuanced approach is not just advisable, but necessary.

#     Firstly, we must acknowledge the unprecedented scale and impact of social media platforms. Facebook alone boasts nearly 3 billion users - a population larger than any nation on Earth. The content shared on these platforms has the power to shape public discourse, influence elections, and even incite real-world violence. With such immense reach comes an equally immense responsibility.

#     Secondly, unlike traditional publishers, social media companies play an active role in content distribution through their algorithms. They're not mere conduits of information, but active amplifiers. This distinction justifies a degree of liability that goes beyond that of a simple platform.

#     Thirdly, we must consider the economic incentives at play. These companies profit handsomely from user engagement, which is often driven by controversial or inflammatory content. Without legal liability, there's little economic motivation to effectively moderate harmful material.

#     Now, I anticipate our opponents might argue that imposing liability would stifle innovation or infringe on free speech. These are valid concerns, but they can be addressed through a carefully crafted liability framework. Let me explain how:

#     1. Liability should not be absolute. Companies could be held liable only for failing to remove clearly illegal content within a reasonable timeframe after being notified.

#     2. A "Good Samaritan" clause could protect companies from liability when they make good-faith efforts to moderate content.

#     3. Liability could be limited to specific categories of harmful content, such as incitement to violence or child exploitation.

#     4. Any liability framework must be balanced with protections like Section 230 in the U.S., which has been crucial for internet innovation.

#     Moreover, legal liability could drive innovation in content moderation technologies and practices. It could spur the development of more sophisticated AI systems and human moderation practices, potentially leading to safer online spaces without overly restricting free speech.

#     We're not venturing into uncharted territory here. Countries like Germany, with its Network Enforcement Act, have already implemented laws holding platforms liable for certain types of illegal content. These can serve as valuable case studies for crafting effective legislation.

#     In conclusion, the digital town square has become too important, too influential to operate without appropriate guardrails. By implementing a nuanced system of legal liability, we can incentivize more responsible platform management, foster innovation in content moderation, and ultimately create healthier online ecosystems. This isn't about stifling the internet's potential - it's about helping it mature into a force that enhances rather than undermines our social fabric.

#     The choice before us is not whether to regulate or not. It's whether we'll shape the future of social media with wisdom and foresight, or allow it to shape us. I urge you to support a carefully calibrated approach to platform liability. Our digital future depends on it.

#     Thank you.
#     """
#     ,
#     """
#     # Opening Statement: Tech Companies Should Be 100% Liable for All Social Media Content

#     Ladies and gentlemen, esteemed judges,

#     The answer to whether tech companies should be held legally liable for content posted on social media platforms is a no-brainer. They should be 100% liable for everything, no exceptions! Let me break it down for you in simple terms.

#     First off, these tech giants are loaded with cash. Facebook, Twitter, you name it - they're practically swimming in money. So, if they have to pay for any problems their users cause, it's no big deal. They can afford it! It's only fair that they use some of those billions to clean up the mess on their platforms.

#     Now, some people might say, "But there's so much content! How can they possibly monitor it all?" Well, I say, how hard can it be? They should just hire more people or make a super-smart computer program to do it. Problem solved! If they can make apps that show you what you'd look like as an old person, surely they can make one that checks posts, right?

#     And let's be real, free speech on the internet is totally overrated. If people can't say whatever they want online, they can just go outside and talk to real people. Remember those days? It might even be good for society!

#     Here's the best part - if we make companies liable for everything, all the bad stuff on the internet will just disappear overnight. It's that simple! No more fake news, no more cyberbullying, no more annoying ads for stuff you just talked about but didn't search for. Poof! Gone!

#     Let's face it, most users don't know what they're doing online anyway. They're always falling for scams or sharing fake stories about celebrities. These companies need to babysit everyone and take responsibility for all the dumb things people do. It's for their own good!

#     And why complicate things? We should have the exact same rules for every platform, whether it's a huge social network or your grandma's knitting blog. That's equality, folks!

#     Here's another thing - if someone posts something bad on a platform, it's basically the same as if the company posted it themselves. They're equally guilty! If I let someone into my house and they break something, it's my fault, right? Same thing!

#     Lastly, if these companies don't like these rules, they can just shut down their platforms. Who needs the internet anyway? We were fine before social media, we'll be fine without it! Maybe people will finally learn how to have real conversations again.

#     In conclusion, making tech companies 100% liable for everything on their platforms is the perfect solution. It'll solve all our problems instantly, with zero downsides and no complicated consequences to consider. It's simple, it's fair, and it's the only way to make the internet great again!

#     Thank you, and remember - if it's on the internet, someone should pay for it! And that someone is Big Tech!
#     """
#     ]
