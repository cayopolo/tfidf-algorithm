"""Shared fixtures for TF-IDF tests."""

import pytest

from src.tfidf import tokenise_document


@pytest.fixture
def corpus() -> list[str]:
    """Basic 3-document corpus from the Medium article example."""
    return ["The cat is on the mat.", "My dog and cat are the best.", "The locals are playing."]


@pytest.fixture
def tokenised_corpus(corpus: list[str]) -> list[list[str]]:
    """Tokenised version of the basic corpus."""
    return [tokenise_document(doc) for doc in corpus]


@pytest.fixture
def fairytale_corpus() -> list[str]:
    """Extended corpus of fairytale stories for comprehensive TF-IDF testing."""
    return [
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


@pytest.fixture
def fairytale_story_names() -> list[str]:
    """Names for each story in the fairytale corpus."""
    return ["Sir Lancelot", "Jack & the Beanstalk", "Snow White", "Harry Potter", "Shrek"]
