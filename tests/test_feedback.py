from app.feedback import analyze

def test_analyze_repeats():
    transcript = "hello hello hello world"
    result = analyze(transcript)
    assert 'repeated_words' in result
    assert result['repeated_words'].get('hello') == 3

def test_analyze_grammar():
    transcript = "this is me i am here"
    result = analyze(transcript)
    assert 'grammar_issues' in result
    assert "Consider capitalizing 'I'" in result['grammar_issues']
