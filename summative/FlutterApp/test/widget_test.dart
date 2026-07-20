import 'package:flutter_test/flutter_test.dart';
import 'package:life_expectancy_app/main.dart';

void main() {
  testWidgets('App renders title and Predict button', (WidgetTester tester) async {
    await tester.pumpWidget(const LifeExpectancyApp());

    expect(find.text('Life Expectancy Predictor'), findsWidgets);
    expect(find.text('Predict'), findsOneWidget);
  });
}