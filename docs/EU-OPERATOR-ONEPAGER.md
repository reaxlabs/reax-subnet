# EU operators: what changes (one page) / Was sich ändert (eine Seite)

## English

**What changes**
- No more bit-identity. The strict self-check (`reax-selfcheck` PASS at >= 99 % simulated rounds) is no longer the entry bar.
  Instead we run an **answer test**: do your engine's answers match our reference answers closely enough, under real load?
- **You choose the parallelism your hardware can carry.** The test ramps concurrency (1, 2, 4, 8, 12, 16, 24, 32 requests at
  once) and tells you the highest level at which the answers stay good. That level becomes your concurrency per GPU
  (no more fixed "1 + 1").
- **Any engine is fine** (our REAX engine, vLLM with the plugin, SGLang, ...) as long as the answers pass. Engine image,
  GPU and driver are noted in the report.
- Using the same GPUs for TrustedTokens stays allowed, as in the revised sub-processing text.

**How the test runs**
1. Quick read, whenever you like: `reax-answertest --model <model> --url <your runtime> --token-file <file> --duration 20`.
   Takes about 10-20 minutes; prints a table and the concurrency you would be admitted at.
2. Binding run: we run it against your endpoint with test traffic only (a public golden set, no customer data), with two
   cold starts (we ask you to restart the runtime once). The same kind of bar as our own nodes (exact numbers after our calibration run).
3. We repeat it when the engine image, GPU model, driver major version or the concurrency changes. During operation we
   spot-check about 2 % of real answers on our reference; if answers drift, we can lower the concurrency (planned as an
   automatic step) and you can re-test to raise it. For that check we store only technical numbers, no request text.

**What stays**
- Contract and pay unchanged: you are paid per verified GPU-hour (hour passes with >= 90 % positive validator rounds, no
  strike, >= 25 decisions/s); pay numbers unchanged.
- Data rules unchanged: processing only inside the EU on the agreed hardware, no logging or storing of request content,
  sub-processing terms as signed.

## Deutsch

**Was sich ändert**
- Keine Bit-Identität mehr. Der strenge Self-Check (`reax-selfcheck` PASS bei >= 99 % simulierten Runden) ist nicht mehr die
  Eintrittshürde. Stattdessen machen wir einen **Antworttest**: Stimmen die Antworten Ihrer Engine unter echter Last gut genug
  mit unseren Referenzantworten überein?
- **Sie wählen die Parallelität, die Ihre Hardware trägt.** Der Test steigert die Last (1, 2, 4, 8, 12, 16, 24, 32 gleichzeitige
  Anfragen) und nennt die höchste Stufe, bei der die Antworten gut bleiben. Diese Stufe wird Ihre Parallelität pro GPU
  (nicht mehr fest "1 + 1").
- **Jede Engine ist erlaubt** (unsere REAX-Engine, vLLM mit Plugin, SGLang, ...), solange die Antworten den Test bestehen.
  Engine-Image, GPU und Treiber stehen im Bericht.
- Die Mitnutzung der GPUs für TrustedTokens bleibt erlaubt, wie im überarbeiteten Unterauftragsverarbeitungs-Text.

**Wie der Test läuft**
1. Schnelltest, jederzeit: `reax-answertest --model <Modell> --url <Ihre Runtime> --token-file <Datei> --duration 20`.
   Dauert etwa 10-20 Minuten; zeigt eine Tabelle und die Stufe, mit der Sie zugelassen würden.
2. Verbindlicher Lauf: wir führen ihn gegen Ihren Endpunkt aus, nur mit Testverkehr (öffentliches Testset, keine Kundendaten),
   mit zwei Kaltstarts (wir bitten Sie einmal, die Runtime neu zu starten). Gleiche Messlatte wie bei unseren eigenen Knoten.
3. Wir wiederholen ihn bei neuem Engine-Image, anderem GPU-Modell, anderer Treiber-Hauptversion oder höherer Parallelität.
   Im Betrieb prüfen wir etwa 2 % der echten Antworten stichprobenartig gegen unsere Referenz; driften die Antworten, können wir die
   Parallelität senken (geplant als automatischer Schritt), und ein neuer Test hebt sie wieder an. Dafür speichern wir nur technische Zahlen, keinen
   Anfragetext.

**Was bleibt**
- Vertrag und Vergütung unverändert: Bezahlung pro verifizierter GPU-Stunde (Stunde zählt bei >= 90 % positiven
  Validator-Runden, kein Strike, >= 25 Entscheidungen/s); Vergütungszahlen unverändert.
- Datenregeln unverändert: Verarbeitung nur in der EU auf der vereinbarten Hardware, keine Protokollierung oder Speicherung von
  Anfrageinhalten, Unterauftragsverarbeitung wie unterschrieben.
