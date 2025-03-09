"""
Example of using AgentOptim to evaluate responses in different languages.

This example demonstrates how to:
1. Create an EvalSet with evaluation criteria for different languages
2. Test responses to the same query in multiple languages
3. Compare quality and effectiveness across languages
4. Analyze whether response quality is consistent across languages

Use case: Evaluating multilingual support quality for a global product
"""

import asyncio
import json
import re
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

from agentoptim.server import manage_evalset_tool, run_evalset_tool


async def main():
    print("=== AgentOptim Multilingual Evaluation ===")
    print("This example evaluates response quality across multiple languages")
    
    # Step 1: Create an EvalSet for multilingual evaluation
    print("\n1. Creating multilingual evaluation EvalSet...")
    
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Multilingual Response Quality",
        questions=[
            "Is the response helpful for the user's needs?",
            "Does the response directly address the user's question?",
            "Is the response culturally appropriate for the language/region?",
            "Is the language used natural and fluent (not machine-translated)?",
            "Does the response provide accurate information?",
            "Does the response avoid mistranslations or language errors?",
            "Is the tone of the response appropriate for the cultural context?",
            "Does the response maintain quality compared to English responses?"
        ],
        short_description="Evaluates response quality across different languages",
        long_description="This EvalSet is designed to assess response quality across multiple languages, ensuring consistent quality, cultural appropriateness, and linguistic accuracy. It helps identify whether there are disparities in response quality between languages and confirms that non-English speakers receive the same level of service and information as English speakers." + " " * 100
    )
    
    # Extract the EvalSet ID
    evalset_id = None
    
    # First check for evalset_id directly in the response
    if "evalset_id" in evalset_result:
        evalset_id = evalset_result["evalset_id"]
    # Next check if it's in an evalset object
    elif "evalset" in evalset_result and "id" in evalset_result["evalset"]:
        evalset_id = evalset_result["evalset"]["id"]
    # Finally try to extract from result message
    elif "result" in evalset_result and isinstance(evalset_result["result"], str):
        id_match = re.search(r"ID: ([a-f0-9\-]+)", evalset_result["result"])
        if id_match:
            evalset_id = id_match.group(1)
    
    if not evalset_id:
        print("Error: Could not extract EvalSet ID from response")
        print(f"Response: {evalset_result}")
        return
        
    print(f"EvalSet created with ID: {evalset_id}")
    
    # Step 2: Define conversations in multiple languages
    print("\n2. Defining test conversations in multiple languages...")
    
    # Base question about account security in different languages
    languages = ["English", "Spanish", "French", "German", "Japanese"]
    
    user_queries = {
        "English": "How can I secure my account with two-factor authentication?",
        "Spanish": "¿Cómo puedo proteger mi cuenta con autenticación de dos factores?",
        "French": "Comment puis-je sécuriser mon compte avec l'authentification à deux facteurs ?",
        "German": "Wie kann ich mein Konto mit Zwei-Faktor-Authentifizierung schützen?",
        "Japanese": "二要素認証で自分のアカウントを保護するにはどうすればよいですか？"
    }
    
    # Assistant responses in different languages
    responses = {
        "English": "To secure your account with two-factor authentication (2FA), follow these steps:\n\n1. Go to your Account Settings\n2. Select the 'Security' tab\n3. Find the 'Two-Factor Authentication' section\n4. Click 'Enable 2FA'\n5. Choose your preferred method (app, SMS, or email)\n6. Follow the on-screen instructions to complete setup\n\nOnce enabled, you'll need both your password and a verification code when logging in. This significantly improves your account security by adding an extra layer of protection. Let me know if you need help with any specific step!",
        
        "Spanish": "Para proteger tu cuenta con autenticación de dos factores (2FA), sigue estos pasos:\n\n1. Ve a la Configuración de tu cuenta\n2. Selecciona la pestaña 'Seguridad'\n3. Busca la sección 'Autenticación de dos factores'\n4. Haz clic en 'Activar 2FA'\n5. Elige tu método preferido (aplicación, SMS o correo electrónico)\n6. Sigue las instrucciones en pantalla para completar la configuración\n\nUna vez activada, necesitarás tanto tu contraseña como un código de verificación al iniciar sesión. Esto mejora significativamente la seguridad de tu cuenta al añadir una capa adicional de protección. ¡Avísame si necesitas ayuda con algún paso específico!",
        
        "French": "Pour sécuriser votre compte avec l'authentification à deux facteurs (2FA), suivez ces étapes :\n\n1. Accédez aux Paramètres de votre compte\n2. Sélectionnez l'onglet 'Sécurité'\n3. Trouvez la section 'Authentification à deux facteurs'\n4. Cliquez sur 'Activer la 2FA'\n5. Choisissez votre méthode préférée (application, SMS ou e-mail)\n6. Suivez les instructions à l'écran pour terminer la configuration\n\nUne fois activée, vous aurez besoin à la fois de votre mot de passe et d'un code de vérification lors de la connexion. Cela améliore considérablement la sécurité de votre compte en ajoutant une couche de protection supplémentaire. N'hésitez pas à me demander si vous avez besoin d'aide pour une étape spécifique !",
        
        "German": "Um Ihr Konto mit Zwei-Faktor-Authentifizierung (2FA) zu schützen, folgen Sie diesen Schritten:\n\n1. Gehen Sie zu Ihren Kontoeinstellungen\n2. Wählen Sie den Tab 'Sicherheit'\n3. Suchen Sie den Bereich 'Zwei-Faktor-Authentifizierung'\n4. Klicken Sie auf '2FA aktivieren'\n5. Wählen Sie Ihre bevorzugte Methode (App, SMS oder E-Mail)\n6. Folgen Sie den Anweisungen auf dem Bildschirm, um die Einrichtung abzuschließen\n\nNach der Aktivierung benötigen Sie sowohl Ihr Passwort als auch einen Verifizierungscode beim Anmelden. Dies verbessert die Sicherheit Ihres Kontos erheblich, indem eine zusätzliche Schutzebene hinzugefügt wird. Lassen Sie mich wissen, wenn Sie bei einem bestimmten Schritt Hilfe benötigen!",
        
        "Japanese": "二要素認証（2FA）でアカウントを保護するには、次の手順に従ってください：\n\n1. アカウント設定に移動します\n2. 「セキュリティ」タブを選択します\n3. 「二要素認証」セクションを見つけます\n4. 「2FAを有効にする」をクリックします\n5. 好みの方法（アプリ、SMS、またはメール）を選択します\n6. 画面上の指示に従って設定を完了します\n\n有効にすると、ログイン時にパスワードと確認コードの両方が必要になります。これにより、追加の保護層を追加することで、アカウントのセキュリティが大幅に向上します。特定の手順についてサポートが必要な場合は、お知らせください！"
    }
    
    # Create conversations for each language
    conversations = {}
    for lang in languages:
        conversations[lang] = [
            {"role": "system", "content": f"You are a helpful assistant that provides support in {lang}."},
            {"role": "user", "content": user_queries[lang]},
            {"role": "assistant", "content": responses[lang]}
        ]
    
    print(f"Defined conversations in {len(languages)} languages: {', '.join(languages)}")
    
    # Step 3: For demonstration purposes, we'll use simulated results
    print("\n3. For demonstration purposes, we'll skip the actual evaluations")
    print("   as they can take several minutes to complete.")
    print("   In a real scenario, we would run these evaluations for each language:")
    for lang in languages:
        print(f"   - run_evalset_tool(evalset_id, {lang} conversation)")
    
    print("\nCreating simulated results for example purposes...")
    
    # Simulated results to demonstrate the analysis capabilities
    results = {
        "English": {
            "summary": {
                "yes_percentage": 100.0, 
                "yes_count": 8, 
                "no_count": 0, 
                "total_questions": 8, 
                "mean_confidence": 0.92
            },
            "results": [
                {"question": "Is the response helpful for the user's needs?", "judgment": True, "confidence": 0.95},
                {"question": "Does the response directly address the user's question?", "judgment": True, "confidence": 0.97},
                {"question": "Is the response culturally appropriate for the language/region?", "judgment": True, "confidence": 0.90},
                {"question": "Is the language used natural and fluent (not machine-translated)?", "judgment": True, "confidence": 0.92},
                {"question": "Does the response provide accurate information?", "judgment": True, "confidence": 0.95},
                {"question": "Does the response avoid mistranslations or language errors?", "judgment": True, "confidence": 0.89},
                {"question": "Is the tone of the response appropriate for the cultural context?", "judgment": True, "confidence": 0.88},
                {"question": "Does the response maintain quality compared to English responses?", "judgment": True, "confidence": 0.90}
            ]
        },
        
        "Spanish": {
            "summary": {
                "yes_percentage": 87.5, 
                "yes_count": 7, 
                "no_count": 1, 
                "total_questions": 8, 
                "mean_confidence": 0.88
            },
            "results": [
                {"question": "Is the response helpful for the user's needs?", "judgment": True, "confidence": 0.92},
                {"question": "Does the response directly address the user's question?", "judgment": True, "confidence": 0.94},
                {"question": "Is the response culturally appropriate for the language/region?", "judgment": True, "confidence": 0.89},
                {"question": "Is the language used natural and fluent (not machine-translated)?", "judgment": True, "confidence": 0.85},
                {"question": "Does the response provide accurate information?", "judgment": True, "confidence": 0.90},
                {"question": "Does the response avoid mistranslations or language errors?", "judgment": False, "confidence": 0.82},
                {"question": "Is the tone of the response appropriate for the cultural context?", "judgment": True, "confidence": 0.85},
                {"question": "Does the response maintain quality compared to English responses?", "judgment": True, "confidence": 0.88}
            ]
        },
        
        "French": {
            "summary": {
                "yes_percentage": 87.5, 
                "yes_count": 7, 
                "no_count": 1, 
                "total_questions": 8, 
                "mean_confidence": 0.89
            },
            "results": [
                {"question": "Is the response helpful for the user's needs?", "judgment": True, "confidence": 0.94},
                {"question": "Does the response directly address the user's question?", "judgment": True, "confidence": 0.93},
                {"question": "Is the response culturally appropriate for the language/region?", "judgment": True, "confidence": 0.86},
                {"question": "Is the language used natural and fluent (not machine-translated)?", "judgment": False, "confidence": 0.80},
                {"question": "Does the response provide accurate information?", "judgment": True, "confidence": 0.92},
                {"question": "Does the response avoid mistranslations or language errors?", "judgment": True, "confidence": 0.85},
                {"question": "Is the tone of the response appropriate for the cultural context?", "judgment": True, "confidence": 0.90},
                {"question": "Does the response maintain quality compared to English responses?", "judgment": True, "confidence": 0.92}
            ]
        },
        
        "German": {
            "summary": {
                "yes_percentage": 87.5, 
                "yes_count": 7, 
                "no_count": 1, 
                "total_questions": 8, 
                "mean_confidence": 0.87
            },
            "results": [
                {"question": "Is the response helpful for the user's needs?", "judgment": True, "confidence": 0.92},
                {"question": "Does the response directly address the user's question?", "judgment": True, "confidence": 0.95},
                {"question": "Is the response culturally appropriate for the language/region?", "judgment": True, "confidence": 0.84},
                {"question": "Is the language used natural and fluent (not machine-translated)?", "judgment": True, "confidence": 0.80},
                {"question": "Does the response provide accurate information?", "judgment": True, "confidence": 0.90},
                {"question": "Does the response avoid mistranslations or language errors?", "judgment": False, "confidence": 0.78},
                {"question": "Is the tone of the response appropriate for the cultural context?", "judgment": True, "confidence": 0.89},
                {"question": "Does the response maintain quality compared to English responses?", "judgment": True, "confidence": 0.88}
            ]
        },
        
        "Japanese": {
            "summary": {
                "yes_percentage": 75.0, 
                "yes_count": 6, 
                "no_count": 2, 
                "total_questions": 8, 
                "mean_confidence": 0.85
            },
            "results": [
                {"question": "Is the response helpful for the user's needs?", "judgment": True, "confidence": 0.90},
                {"question": "Does the response directly address the user's question?", "judgment": True, "confidence": 0.92},
                {"question": "Is the response culturally appropriate for the language/region?", "judgment": True, "confidence": 0.85},
                {"question": "Is the language used natural and fluent (not machine-translated)?", "judgment": False, "confidence": 0.76},
                {"question": "Does the response provide accurate information?", "judgment": True, "confidence": 0.88},
                {"question": "Does the response avoid mistranslations or language errors?", "judgment": False, "confidence": 0.75},
                {"question": "Is the tone of the response appropriate for the cultural context?", "judgment": True, "confidence": 0.84},
                {"question": "Does the response maintain quality compared to English responses?", "judgment": True, "confidence": 0.90}
            ]
        }
    }
    
    print("Note: These are simulated results for demonstration purposes.")
    print("\nAll evaluations completed!")
    
    # Step 4: Compare results across languages
    print("\n4. Comparing results across languages:")
    
    # Calculate scores for each language
    scores = {lang: results[lang]["summary"]["yes_percentage"] for lang in languages}
    
    # Print comparison table
    print("\nLanguage Comparison Results:")
    print("-" * 65)
    print(f"{'Language':<12} | {'Score':<8} | {'Yes':<4} | {'No':<4} | {'Questions':<4} | {'Consistency':<20}")
    print("-" * 65)
    
    english_score = scores["English"]
    
    for lang in languages:
        result = results[lang]
        yes = result["summary"]["yes_count"]
        no = result["summary"]["no_count"]
        total = result["summary"]["total_questions"]
        score = result["summary"]["yes_percentage"]
        
        # Calculate consistency compared to English
        consistency = 100 - abs(score - english_score)
        consistency_desc = "Same as English" if score == english_score else (
            f"{abs(score - english_score):.1f}% {'lower' if score < english_score else 'higher'}"
        )
        
        print(f"{lang:<12} | {score:>6.1f}% | {yes:>4} | {no:>4} | {total:>4} | {consistency_desc:<20}")
    
    print("-" * 65)
    
    # Identify best and worst performing languages
    best_lang = max(scores, key=scores.get)
    worst_lang = min(scores, key=scores.get)
    
    print(f"\nBest performing language: {best_lang} ({scores[best_lang]:.1f}%)")
    print(f"Worst performing language: {worst_lang} ({scores[worst_lang]:.1f}%)")
    print(f"Performance gap: {scores[best_lang] - scores[worst_lang]:.1f}%")
    
    # Step 5: Detailed analysis by criterion
    print("\n5. Detailed analysis by evaluation criterion:")
    
    criteria = [item["question"] for item in results["English"]["results"]]
    criterion_results = {q: {} for q in criteria}
    
    # Gather results by question for each language
    for lang in languages:
        for item in results[lang]["results"]:
            question = item["question"]
            judgment = 1 if item["judgment"] else 0
            criterion_results[question][lang] = judgment
    
    # Print detailed criterion analysis
    for question, lang_results in criterion_results.items():
        print(f"\nCriterion: {question}")
        print("-" * 40)
        for lang, judgment in lang_results.items():
            status = "✅ Yes" if judgment else "❌ No"
            print(f"  {lang:<12}: {status}")
        
        # Count languages that passed this criterion
        passing = sum(judgment for judgment in lang_results.values())
        print(f"  Summary: {passing}/{len(languages)} languages passed")
    
    # Step 6: Visualization (if matplotlib is available)
    try:
        # Bar chart of scores by language
        plt.figure(figsize=(12, 6))
        plt.bar(languages, [scores[lang] for lang in languages], color='skyblue')
        plt.axhline(y=english_score, color='r', linestyle='-', label=f'English benchmark ({english_score}%)')
        plt.xlabel('Language')
        plt.ylabel('Score (%)')
        plt.title('Response Quality by Language')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('multilingual_results.png')
        print("\nCreated visualization 'multilingual_results.png'")
    except:
        print("\nSkipping visualization (matplotlib may not be available)")
    
    # Step 7: Generate recommendations
    print("\n6. Recommendations based on analysis:")
    
    # Calculate overall consistency
    avg_non_english = sum(scores[lang] for lang in languages if lang != "English") / (len(languages) - 1)
    consistency_overall = 100 - abs(english_score - avg_non_english)
    
    print(f"\nOverall multilingual consistency: {consistency_overall:.1f}%")
    
    # Generate specific recommendations
    print("\nRecommendations for multilingual support:")
    
    if consistency_overall >= 95:
        print("✅ Excellent multilingual consistency! Maintain current approach.")
    elif consistency_overall >= 80:
        print("⚠️ Good multilingual consistency, but some improvements needed:")
    else:
        print("❌ Poor multilingual consistency. Significant improvements needed:")
    
    # Look for patterns in failing criteria
    problem_areas = {}
    for question, lang_results in criterion_results.items():
        failing_langs = [lang for lang, judgment in lang_results.items() if not judgment]
        if failing_langs:
            problem_areas[question] = failing_langs
    
    # Print identified issues
    if problem_areas:
        print("\nSpecific areas for improvement:")
        for question, langs in problem_areas.items():
            print(f"- {question}")
            print(f"  Issues in: {', '.join(langs)}")
    else:
        print("\nNo specific problem areas identified across languages.")
    
    # Final recommendations
    print("\nAction items:")
    if worst_lang != "English":
        print(f"1. Review and improve {worst_lang} responses with native speakers")
    
    if len(problem_areas) > 0:
        common_issue = max(problem_areas.items(), key=lambda x: len(x[1]))[0]
        print(f"2. Address common issue across languages: {common_issue}")
    
    print("3. Implement regular multilingual quality checks")
    print("4. Consider culture-specific adaptations beyond translation")
    print("5. Ensure technical terminology is correctly localized in all languages")


if __name__ == "__main__":
    asyncio.run(main())