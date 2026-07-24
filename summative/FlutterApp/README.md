# Life Expectancy Predictor (Flutter)

Single-page mobile app that sends the 8 health indicators to the prediction API
and displays the predicted life expectancy (or a validation error).

```bash
# run from the summative/ directory

cd FlutterApp
flutter pub get
flutter run            
```

Before running, set your deployed API URL at the top of `lib/main.dart`:

```dart
const String kApiBaseUrl = "https://linear-regression-model-z9ly.onrender.com";
```
