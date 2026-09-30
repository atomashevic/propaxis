"""A small, runnable example with the E5 encoder. Downloads the model on first use."""

from propaxis import Component, Construct, Encoder

TRANSIT = Component(
    "transit_support",
    positive=[
        "The city should invest more in public transportation.",
        "Expanding bus and rail services should be a budget priority.",
        "Affordable and frequent public transit deserves public funding.",
    ],
    contrast=[
        "The city should reduce spending on public transportation.",
        "Expanding bus and rail services is not a budget priority.",
        "Public transit should receive less public funding.",
    ],
)

# Small teaching sample. Use a larger, representative dataset for research.
REFERENCE = [
    "We should add bus routes to underserved neighborhoods.",
    "I support a larger budget for trains and buses.",
    "Affordable transit makes it easier for people to get to work.",
    "The council should fund more frequent bus services.",
    "Our transport budget should focus on maintaining roads.",
    "We cannot afford another public rail expansion.",
    "Bus subsidies should be reduced next year.",
    "Public transportation already receives too much funding.",
    "The council will meet on Tuesday to discuss the budget.",
    "The transport committee published its annual report.",
    "Residents disagree about the planned bus route.",
    "The city is reviewing several transport proposals.",
]


def main():
    construct = Construct("transit_stance", [TRANSIT])
    instrument = construct.fit(REFERENCE, encoder=Encoder("e5-small-v2"))
    result = instrument.score(
        [
            "We need more frequent buses, and the city should pay for them.",
            "The city should stop subsidizing public transportation.",
        ]
    )
    for row in result.scores:
        print(row)
    print("Model:", instrument.encoder_config)
    print("Evidence:", result.evidence[0])


if __name__ == "__main__":
    main()
