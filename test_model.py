import requests
import json
import time

# Test cases
test_cases = [
    {
        "name": "Fake - Political",
        "text": "BREAKING: Hillary Clinton just got INDICTED! Federal investigators have uncovered evidence that Hillary Clinton sent classified information through her private email server while she was Secretary of State. The FBI has recommended criminal charges, and she could face up to 10 years in prison. Mainstream media is hiding this story!",
        "expected": "FAKE"
    },
    {
        "name": "Fake - Health", 
        "text": "SHOCKING: Doctors don't want you to know that drinking baking soda mixed with lemon juice can cure cancer in just 3 days! Big Pharma is suppressing this natural remedy because they want to sell expensive chemotherapy drugs. Thousands of people have already been cured!",
        "expected": "FAKE"
    },
    {
        "name": "Real - Politics",
        "text": "President Biden signed the Infrastructure Investment and Jobs Act into law today, a $1.2 trillion bipartisan package that will fund improvements to roads, bridges, public transit, and broadband internet access across the United States.",
        "expected": "REAL"
    },
    {
        "name": "Real - Science",
        "text": "NASA's Perseverance rover has discovered organic molecules on Mars, providing new evidence that the Red Planet may have once supported microbial life. The rover has been collecting samples and analyzing the Martian surface for signs of past life.",
        "expected": "REAL"
    },
    {
        "name": "Short Text",
        "text": "Scientists discovered a new planet.",
        "expected": "UNCERTAIN"
    }
]

print("=" * 60)
print("🧪 TESTING FAKE NEWS DETECTION SYSTEM")
print("=" * 60)

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. Testing: {test['name']}")
    print(f"   Expected: {test['expected']}")
    print(f"   Text: {test['text'][:80]}...")
    
    try:
        response = requests.post(
            'http://127.0.0.1:5000/predict',
            json={'news_text': test['text']},
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   📊 Result: {result['result']}")
            print(f"   📈 Confidence: {result['confidence']}%")
            print(f"   📉 Fake: {result['fake_probability']}% | Real: {result['real_probability']}%")
            
            # Check if matches expectation
            if test['expected'] != "UNCERTAIN":
                if result['result'] == test['expected']:
                    print("   ✅ PASS")
                else:
                    print(f"   ❌ FAIL (Expected {test['expected']})")
            else:
                print("   ⚠️ UNCERTAIN (Acceptable any result)")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        print("   Make sure server is running: python app.py")
    
    time.sleep(0.5)  # Small delay between requests

print("\n" + "=" * 60)
print("✅ TESTING COMPLETE")
print("=" * 60)