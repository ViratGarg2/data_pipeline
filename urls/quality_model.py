#!/usr/bin/env python3
"""
Interactive tool to test the FastText quality classifier model.

Usage:
    # Interactive mode - enter texts manually
    python test_quality_model.py --model quality_classifier.bin
    
    # Test with a specific text
    python test_quality_model.py --model quality_classifier.bin --text "Your text here"
    
    # Test with texts from a file (one per line)
    python test_quality_model.py --model quality_classifier.bin --file texts_to_test.txt
    
    # Run built-in test examples
    python test_quality_model.py --model quality_classifier.bin --examples
"""

import argparse
from pathlib import Path

import fasttext


def load_model(model_path: str) -> fasttext.FastText._FastText:
    """Load the FastText model from disk."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = fasttext.load_model(model_path)
    print(f"Model loaded successfully!")
    print(f"  Labels: {model.labels}")
    print()
    return model


def predict_quality(model: fasttext.FastText._FastText, text: str, verbose: bool = True) -> tuple[str, float]:
    """
    Predict the quality of a text.
    
    Args:
        model: Loaded FastText model
        text: Text to classify
        verbose: Whether to print results
        
    Returns:
        Tuple of (label, confidence)
    """
    # Clean text for prediction (single line)
    clean_text = text.replace("\n", " ").replace("\r", " ")
    clean_text = " ".join(clean_text.split())  # Collapse whitespace
    
    # Get prediction
    labels, probs = model.predict(clean_text, k=2)  # Get top 2 predictions
    
    primary_label = labels[0].replace("__label__", "")
    primary_prob = probs[0]
    
    if verbose:
        print("-" * 70)
        print(f"TEXT: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"\nPREDICTION: {primary_label.upper()}")
        print(f"CONFIDENCE: {primary_prob:.4f} ({primary_prob:.2%})")
        
        if len(labels) > 1:
            alt_label = labels[1].replace("__label__", "")
            alt_prob = probs[1]
            print(f"\nAlternative: {alt_label} ({alt_prob:.4f})")
        
        # Quality interpretation
        if primary_label == "positive":
            print(f"\n✅ HIGH QUALITY - This text appears to be well-written/informative")
        else:
            print(f"\n❌ LOW QUALITY - This text appears to be low quality/spam-like")
        print("-" * 70)
    
    return primary_label, primary_prob


def run_interactive(model: fasttext.FastText._FastText):
    """Run interactive mode where user can input texts."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("Enter text to classify (or 'quit' to exit, 'multiline' for multi-line input)")
    print("=" * 70 + "\n")
    
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'multiline':
                print("Enter multiple lines (type 'END' on a new line when done):")
                lines = []
                while True:
                    line = input()
                    if line.strip().upper() == 'END':
                        break
                    lines.append(line)
                user_input = "\n".join(lines)
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            predict_quality(model, user_input)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


def run_examples(model: fasttext.FastText._FastText):
    """Run prediction on built-in example texts."""
    examples = [
        # High quality examples
        (
            """Speak Korean Now!
Teach English Abroad and Get Paid to see the World!
Korean Job Discussion Forums Forum Index Korean Job Discussion Forums
\"The Internet's Meeting Place for ESL/EFL Teachers from Around the World!\"
 FAQFAQ   SearchSearch   MemberlistMemberlist   UsergroupsUsergroups   RegisterRegister 
 ProfileProfile   Log in to check your private messagesLog in to check your private messages   Log inLog in 
This page is maintained by the one and only Dave Sperling.
Contact Dave's ESL Cafe
Copyright © 2018 Dave Sperling. All Rights Reserved.
Powered by phpBB © 2001, 2002 phpBB Group
TEFL International Supports Dave's ESL Cafe
TEFL Courses, TESOL Course, English Teaching Jobs - TEFL International
"""
        ),
        (
            "Machine learning is a subset of artificial intelligence that provides systems "
            "the ability to automatically learn and improve from experience without being "
            "explicitly programmed. Machine learning focuses on the development of computer "
            "programs that can access data and use it to learn for themselves."
        ),
        (
            """Anarchism
First published Tue Oct 3, 2017; substantive revision Tue Oct 26, 2021
Anarchism is a political theory that is skeptical of the justification of authority and power. Anarchism is usually grounded in moral claims about the importance of individual liberty, often conceived as freedom from domination. Anarchists also offer a positive theory of human flourishing, based upon an ideal of equality, community, and non-coercive consensus building. Anarchism has inspired practical efforts at establishing utopian communities, radical and revolutionary political agendas, and various forms of direct action. This entry primarily describes “philosophical anarchism”: it focuses on anarchism as a theoretical idea and not as a form of political activism. While philosophical anarchism describes a skeptical theory of political legitimation, anarchism is also a concept that has been employed in philosophical and literary theory to describe a sort of anti-foundationalism. Philosophical anarchism can mean either a theory of political life that is skeptical of attempts to justify state authority or a philosophical theory that is skeptical of the attempt to assert firm foundations for knowledge.

1. Varieties of Anarchism
There are various forms of anarchism. Uniting this variety is the general critique of centralized, hierarchical power and authority. Given that authority, centralization, and hierarchy show up in various ways and in different discourses, institutions, and practices, it is not surprising that the anarchist critique has been applied in diverse ways.

1.1 Political Anarchism
Anarchism is primarily understood as a skeptical theory of political legitimation. The term anarchism is derived from the negation of the Greek term arché, which means first principle, foundation, or ruling power. Anarchy is thus rule by no one or non-rule. Some argue that non-ruling occurs when there is rule by all—with consensus or unanimity providing an optimistic goal (see Depuis-Déri 2010).

Political anarchists focus their critique on state power, viewing centralized, monopolistic coercive power as illegitimate. Anarchists thus criticize “the state”. Bakunin provides a paradigm historical example, saying:

If there is a State, there must be domination of one class by another and, as a result, slavery; the State without slavery is unthinkable—and this is why we are the enemies of the State. (Bakunin 1873 [1990: 178])

A more recent example comes from Gerard Casey who writes, “states are criminal organizations. All states, not just the obviously totalitarian or repressive ones” (Casey 2012: 1).

Such sweeping generalizations are difficult to support. Thus anarchism as political philosophy faces the challenge of specificity. States have been organized in various ways. Political power is not monolithic. Sovereignty is a complicated matter that includes divisions and distributions of power (see Fiala 2015). Moreover, the historical and ideological context of a given anarchist’s critique makes a difference in the content of the political anarchist’s critique. Bakunin was responding primarily to a Marxist and Hegelian view of the state, offering his critique from within the global socialist movement; Casey is writing in the Twenty-First Century in the era of liberalism and globalization, offering his critique from within the movement of contemporary libertarianism. Some anarchists engage in broad generalizations, aiming for a total critique of political power. Others will present a localized critique of a given political entity. An ongoing challenge for those who would seek to understand anarchism is to realize how historically and ideologically diverse approaches fit under the general anarchist umbrella. We look at political anarchism in detail below.

1.2 Religious Anarchism
The anarchist critique has been extended toward the rejection of non-political centralization and authority. Bakunin extended his critique to include religion, arguing against both God and the State. Bakunin rejected God as the absolute master, saying famously, “if God really existed, it would be necessary to abolish him” (Bakunin 1882 [1970: 28]).

There are, however, religious versions of anarchism, which critique political authority from a standpoint that takes religion seriously. Rapp (2012) has shown how anarchism can be found in Taoism. And Ramnath (2011) has identified anarchist threads in Islamic Sufism, in Hindu bhakti movements, in Sikhism’s anti-caste efforts, and in Buddhism. We consider anarchism in connection with Gandhi below. But we focus here on Christian anarchism.

Christian anarchist theology views the kingdom of God as lying beyond any human principle of structure or order. Christian anarchists offer an anti-clerical critique of ecclesiastical and political power. Tolstoy provides an influential example. Tolstoy claims that Christians have a duty not to obey political power and to refuse to swear allegiance to political authority (see Tolstoy 1894). Tolstoy was also a pacifist. Christian anarcho-pacifism views the state as immoral and unsupportable because of its connection with military power (see Christoyannopoulos 2011). But there are also non-pacifist Christian anarchists. Berdyaev, for example, builds upon Tolstoy and in his own interpretation of Christian theology. Berdyaev concludes: “The Kingdom of God is anarchy” (Berdyaev 1940 [1944: 148]).

Christian anarchists have gone so far as to found separatist communities where they live apart from the structures of the state. Notable examples include New England transcendentalists such as William Garrison and Adin Ballou. These transcendentalists had an influence on Tolstoy (see Perry 1973 [1995]).

Other notable Christians with anarchist sympathies include Peter Maurin and Dorothy Day of the Catholic Worker movement. In more recent years, Christian anarchism has been defended by Jacques Ellul who links Christian anarchism to a broad social critique. In addition to being pacifistic, Ellul says, Christian anarchism should also be “antinationalist, anticapitalist, moral, and antidemocratic” (Ellul 1988 [1991: 13]). The Christian anarchist ought to be committed to “a true overturning of authorities of all kinds” (Ellul 1988 [1991: 14]). When asked whether a Christian anarchist should vote, Ellul says no. He states, “anarchy first implies conscientious objection” (Ellul 1988 [1991: 15]).

1.3 Theoretical Anarchism
Anarchist rejection of authority has application in epistemology and in philosophical and literary theory. One significant usage of the term shows up in American pragmatism. William James described his pragmatist philosophical theory as a kind of anarchism: “A radical pragmatist is a happy-go-lucky anarchistic sort of creature” (James 1907 [1981: 116]). James had anarchist sympathies, connected to a general critique of systematic philosophy (see Fiala 2013b). Pragmatism, like other anti-systematic and post-Hegelian philosophies, gives up on the search for an arché or foundation.

Anarchism thus shows up as a general critique of prevailing methods. An influential example is found in the work of Paul Feyerabend, whose Against Method provides an example of “theoretical anarchism” in epistemology and philosophy of science (Feyerabend 1975 [1993]). Feyerabend explains:

Science is an essentially anarchic enterprise: theoretical anarchism is more humanitarian and more likely to encourage progress than its law-and-order alternatives. (Feyerabend 1975 [1993: 9])

His point is that science ought not be constrained by hierarchically imposed principles and strict rule following.

Post-structuralism and trends in post-modernism and Continental philosophy can also be anarchistic (see May 1994). So-called “post-anarchism” is a decentered and free-flowing discourse that deconstructs power, questions essentialism, and undermines systems of authority. Following upon the deconstructive and critical work of authors such as Derrida, Deleuze, Foucault, and others, this critique of the arché goes all the way down. If there is no arché or foundation, then we are left with a proliferation of possibilities. Emerging trends in globalization, cyber-space, and post-humanism make the anarchist critique of “the state” more complicated, since anarchism’s traditional celebration of liberty and autonomy can be critically scrutinized and deconstructed (see Newman 2016).

Traditional anarchists were primarily interested in sustained and focused political activism that led toward the abolition of the state. The difference between free-flowing post-anarchism and traditional anarchism can be seen in the realm of morality. Anarchism has traditionally been critical of centralized moral authority—but this critique was often based upon fundamental principles and traditional values, such as autonomy or liberty. But post-structuralism—along with critiques articulated by some feminists, critical race theorists, and critics of Eurocentrism—calls these values and principles into question.

1.4 Applied Anarchism
The broad critical framework provided by the anarchist critique of authority provides a useful theory or methodology for social critique. In more recent iterations, anarchism has been used to critique gender hierarchies, racial hierarchies, and the like—also including a critique of human domination over nature. Thus anarchism also includes, to name a few varieties: anarcha-feminism or feminist anarchism (see Kornegger 1975), queer anarchism or anarchist queer theory (see Daring et al. 2010), green anarchism or eco-anarchism also associated with anarchist social ecology (see Bookchin 1971 [1986]), Black and indigenous anarchisms and other anarchist critiques of white supremacy and Eurocentrism (to be discussed below); and even anarcho-veganism or “veganarchism” (see Nocella, White, & Cudworth 2015). In the anarcho-vegan literature we find the following description of a broad and inclusive anarchism:

Anarchism is a socio-political theory which opposes all systems of domination and oppression such as racism, ableism, sexism, anti-LGBTTQIA, ageism, sizeism, government, competition, capitalism, colonialism, imperialism and punitive justice, and promotes direct democracy, collaboration, interdependence, mutual aid, diversity, peace, transformative justice and equity. (Nocella et al. 2015: 7)

A thorough-going anarchism would thus offer a critique of anything and everything that smacks of hierarchy, domination, centralization, and unjustified authority.

Anarchists who share these various commitments often act upon their critique of authority by engaging in nonconformist practices (free love, nudism, gender disruption, and so on) or by forming intentional communities that live “off the grid” and outside of the norms of mainstream culture. In extreme forms this becomes anarcho-primitivism or anti-civilizational anarchism (see Zerzan 2008, 2010; Jensen 2006). Alternative anarchist societies have existed in religious communes in post-Reformation Europe and in the early United States, in Nineteenth Century American utopian communities, the hippy communes of the Twentieth Century, anarchist squats, temporary autonomous zones (see Bey 1985), and occasional gatherings of like-minded people.

Given this sort of antinomianism and non-conformism it is easy to see that anarchism also often includes a radical critique of traditional ethical norms and principles. Thus radical ethical anarchism can be contrasted with what we might call bourgeois anarchism (with radical anarchism seeking to disrupt traditional social norms and bourgeois anarchism seeking freedom from the state that does not seek such disruption). And although some argue that anarchists are deeply ethical—committed to liberty and solidarity—others will argue that anarchists are moral nihilists who reject morality entirely or who at least reject the idea that there could be a single source of moral authority (see essays in Franks & Wilson 2010).

In more recent explorations and applications of anarchist thought, the anarchist critique has been related and connected to a variety of emerging theoretical issues and applied concerns. Hilary Lazar (2018) for example, explores how anarchism connects to intersectionality and issues related to multiculturalism. And Sky Croeser (2019) explores how anarchism is connected to the emerging technologies including the Internet. And there are anarchist elements in the development of shared technological and information, as for example in the development of cryptocurrencies, which create economies that outside of traditional state-based economic systems.

1.5 Black, Indigenous, and Decolonizing Anarchism
As mentioned above, among the varieties of applied anarchism we find anarchism associated with various liberation movements and critiques of white supremacy, Eurocentrism, and colonialism. This could be connected with feminist anarchism, women’s liberation movements, and an anarchist critique of patriarchy. We’ll focus here on the anarchist critique found in Black and indigenous liberation movements. Gandhi’s movement in India could be included here (as discussed below).

One focal point here is a claim about anarchist characteristics thought to be found in the social structures of indigenous peoples. Sometimes this is a romantic projection of anti-civilizational anarchists such as John Zerzan, who echoes Rousseau’s naive and ill-informed ideal of the “noble savage.” One must be careful to avoid essentializing claims made about indigenous cultures and political societies. The Inca and the Aztec empires were obviously not utopian anarchist collectives. Nonetheless, scholars of indigeneity affirm the anarchist critique of dominant hegemonies as part of the effort of liberation that would allow indigenous people a degree of self-determination (see Johnson and Ferguson 2019).

Black and indigenous anarchisms provide a radical critique, which holds that the global history of genocide, slavery, colonization, and exploitation rest upon the assumption of white supremacy. White supremacy is thus understood, from this point of view, as a presupposition of statism, centralization, hierarchy, and authority. The anarchist critique of white supremacy is thus linked to a critique of social and political systems that evolved out of the history of slavery and native genocide to include apartheid, inequality, caste/racial hierarchies, and other forms of structural racism. Some defenders of Black anarchism go so far as to suggest that when “Blackness” is defined in opposition to structures of white supremacy, there is a kind of anarchism woven into the concept. Anderson and Samudzi write,

While bound to the laws of the land, Black America can be understood as an extra-state entity because of Black exclusion from the liberal social contract. Due to this extra-state location, Blackness is, in so many ways, anarchistic. (Anderson and Samudzi 2017: no page numbers)

This implies that the experience of Black people unfolds in a social and political world that its defined by its exclusion from power. A similar implication holds for indigenous people, who have been subjugated and dominated by colonial power. Liberation movements thus spring from a social experience that is in a sense anarchic (i.e., developed in exclusion from and opposition to structures of power). It is not surprising, then, that some liberatory activists espouse and affirm anarchism. The American activist Lorenzo Kom’boa Ervin, for example, affirms anarchism in pursuit of Black liberation (Ervin 1997 [2016]). He explains that Black anarchism is different from what he describes as the more authoritarian hierarchy of the Black Panther party. He also argues against the authoritarian structure of religiously oriented Black liberation movements, such as that led by Martin Luther King, Jr.

A significant issue in Black and indigenous anarchisms is the effort to decolonize anarchism itself. Many of the key figures in the anarchist tradition are white, male, and European. The concerns of anarchists such as Kropotkin or Bakunin may be different from the concerns of African Americans or from the concerns of indigenous people in Latin America or elsewhere around the globe. One solution to this problem is to retrieve forgotten voices from within the tradition. In this regard, we might consider Lucy Parsons (also known as Lucy Gonzalez), a former slave who espoused anarchism. Parsons explained that she affirmed anarchism because the political status quo produced nothing but misery and starvation for the masses of humanity. To resolve this an anarchist revolution was needed. Parsons said,

Most anarchists believe the coming change can only come through a revolution, because the possessing class will not allow a peaceful change to take place; still we are willing to work for peace at any price, except at the price of liberty. (Parsons 1905 [2010])

2. Anarchism in Political Philosophy"""
        ),
        
        # Low quality examples
        (
            "CLICK HERE NOW!!! FREE MONEY BEST DEALS LIMITED TIME OFFER "
            "BUY NOW CHEAP PRICES AMAZING DISCOUNTS!!!"
        ),
        (
            "asdfasdf jkljkl random text spam spam spam click click click "
            "free free free buy now sale sale sale"
        ),
        (
            "Copyright 2024 All Rights Reserved. Privacy Policy. Terms of Service. "
            "Contact Us. About. FAQ. Home. Menu. Login. Register. Search."
        ),
        (
            "Re: Re: Re: FWD: FWD: You won't believe this!!! "
            "Share with 10 friends or bad luck for 7 years!!!"
        ),
        
        # Borderline examples
        (
            "This is a simple sentence. It has basic structure. "
            "The words are common. Nothing special here."
        ),
        (
            "Hello world. This is a test message. "
            "I am writing some text to see how the model performs."
        ),
    ]
    
    print("\n" + "=" * 70)
    print("RUNNING BUILT-IN EXAMPLES")
    print("=" * 70)
    
    results = {"positive": 0, "negative": 0}
    
    for i, text in enumerate(examples, 1):
        print(f"\n[Example {i}/{len(examples)}]")
        label, prob = predict_quality(model, text)
        results[label] += 1
        print()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total examples: {len(examples)}")
    print(f"Classified as POSITIVE (high quality): {results['positive']}")
    print(f"Classified as NEGATIVE (low quality): {results['negative']}")


def test_from_file(model: fasttext.FastText._FastText, file_path: str):
    """Test model on texts from a file (one text per line)."""
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"\nReading texts from: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(texts)} texts to classify\n")
    
    results = {"positive": 0, "negative": 0}
    predictions = []
    
    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}]")
        label, prob = predict_quality(model, text)
        results[label] += 1
        predictions.append((text[:50], label, prob))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total texts: {len(texts)}")
    print(f"Positive (high quality): {results['positive']} ({results['positive']/len(texts):.1%})")
    print(f"Negative (low quality): {results['negative']} ({results['negative']/len(texts):.1%})")
    
    # Show confidence distribution
    print("\n" + "-" * 70)
    print("PREDICTIONS BY CONFIDENCE:")
    print("-" * 70)
    
    sorted_preds = sorted(predictions, key=lambda x: x[2], reverse=True)
    
    print("\nMost confident predictions:")
    for text, label, prob in sorted_preds[:5]:
        print(f"  [{label}] {prob:.3f}: {text}...")
    
    print("\nLeast confident predictions:")
    for text, label, prob in sorted_preds[-5:]:
        print(f"  [{label}] {prob:.3f}: {text}...")


def main():
    parser = argparse.ArgumentParser(
        description="Test FastText quality classifier model"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="quality_classifier.bin",
        help="Path to the FastText model file (default: quality_classifier.bin)"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Single text to classify"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="File with texts to classify (one per line)"
    )
    parser.add_argument(
        "--examples", "-e",
        action="store_true",
        help="Run built-in example texts"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have trained the model first:")
        print("  python train_quality_classifier.py --positive positive.txt --negative negative.txt")
        return 1
    
    # Determine mode
    if args.text:
        # Single text mode
        predict_quality(model, args.text)
    elif args.file:
        # File mode
        test_from_file(model, args.file)
    elif args.examples:
        # Examples mode
        run_examples(model)
    else:
        # Default to interactive mode
        run_interactive(model)
    
    return 0


if __name__ == "__main__":
    exit(main())
