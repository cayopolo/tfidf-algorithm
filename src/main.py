from tfidf import compute_tfidf, tokenise_document

corpus = [
    """Once upon a time, in a faraway land, there was a brave knight named Sir Lancelot. He was known for his
strength, courage, and chivalry. One day, the King of the land asked Sir Lancelot to rescue his daughter, who
had been kidnapped by an evil sorcerer. Sir Lancelot set out on his quest, facing many dangers along the way.
But he never gave up, and eventually he rescued the princess and defeated the sorcerer. The people of the land
cheered for Sir Lancelot, and he became a legend in his own time.""",
    """In a village at the foot of a mountain, there lived a poor farmer named Jack. He had a small farm and a cow,
which was his only possession. One day, the cow stopped giving milk, and Jack didn't know what to do. So he
decided to sell the cow at the market. On the way, he met a stranger who offered to trade five magic beans for
the cow. Jack agreed, and when he got home, his mother was furious. But that night, the magic beans grew
into a giant beanstalk, and Jack climbed it to find a castle in the clouds. There, he met a giant who had a goose
that laid golden eggs. Jack stole the goose and ran down the beanstalk, but the giant followed him. Jack
chopped down the beanstalk, and the giant fell to his death. Jack and his mother lived happily ever after with
the golden eggs.""",
    """In a kingdom ruled by a wicked queen, there lived a beautiful princess named Snow White. The queen was
jealous of Snow White's beauty, and ordered a huntsman to kill her. But the huntsman couldn't do it, so he left
Snow White in the forest. There, she met seven dwarfs who took her in and cared for her. But the queen found
out that Snow White was still alive, and disguised herself as an old woman to give Snow White a poisoned
apple. Snow White fell into a deep sleep, but a prince came and woke her with a kiss. They lived happily ever
after, and the queen got what she deserved.""",
    """In a world of magic and wonder, there was a young wizard named Harry Potter. He had been orphaned as a
baby, and was raised by his cruel relatives. But one day, he received a letter from Hogwarts School of
Witchcraft and Wizardry, inviting him to attend. There, he learned about his true heritage and his destiny to
defeat the dark wizard Voldemort. Harry made many friends at Hogwarts, including Hermione Granger and
Ron Weasley. Together, they faced many challenges and battles, but in the end, Harry was able to vanquish
Voldemort and bring peace to the wizarding world.""",
    """In a land of dragons and knights, there was a beautiful princess named Fiona. She had been cursed by a
wicked sorcerer and turned into an ogre. One day, a brave knight named Shrek was sent to rescue her from a
tower. But when he found her, he discovered that she was an ogre. They didn't get along at first, but eventually
they fell in love. Along the way, they met many fairy tale characters, including a talking donkey and a
gingerbread man. Together, they defeated the evil Lord Farquaad and lived happily ever after in the swamp.""",
]


def main() -> None:
    """Demonstration of TF-IDF analysis on fairy tale corpus"""
    print("Preprocessing corpus...")
    tokenised_corpus = [tokenise_document(doc) for doc in corpus]

    # Analyse each document to find highest and lowest TF-IDF scoring words
    print("=" * 80)
    print("Top 5 Highest and Lowest TF-IDF Scoring Words Per Story")
    print("=" * 80)

    story_names = ["Sir Lancelot", "Jack & the Beanstalk", "Snow White", "Harry Potter", "Shrek"]
    for story_name, tokenised_doc in zip(story_names, tokenised_corpus, strict=False):
        print(f"\n{story_name}")
        print("-" * 80)

        unique_words = set(tokenised_doc)

        # Calculate TF-IDF for each unique word
        word_scores = []
        for word in unique_words:
            score = compute_tfidf(word, tokenised_doc, tokenised_corpus)
            word_scores.append((word, score))

        # Sort by score
        word_scores.sort(key=lambda x: x[1], reverse=True)

        print("\nHighest TF-IDF scores (most distinctive to this story):")
        for word, score in word_scores[:5]:
            print(f"  {word:20s} {score:.4f}")

        print("\nLowest TF-IDF scores (common across corpus):")
        for word, score in word_scores[-5:]:
            print(f"  {word:20s} {score:.4f}")


if __name__ == "__main__":
    main()
