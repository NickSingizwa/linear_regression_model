import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

// Base URL of the deployed prediction API.
const String kApiBaseUrl = "https://linear-regression-model-z9ly.onrender.com";

void main() => runApp(const LifeExpectancyApp());

class LifeExpectancyApp extends StatelessWidget {
  const LifeExpectancyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Life Expectancy Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF2A6F97),
        useMaterial3: true,
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
          isDense: true,
        ),
      ),
      home: const PredictionPage(),
    );
  }
}

/// One input variable: its JSON key, label, valid range and helper hint.
class InputField {
  final String key;
  final String label;
  final double min;
  final double max;
  final String hint;
  final TextEditingController controller = TextEditingController();
  InputField(this.key, this.label, this.min, this.max, this.hint);
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});
  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();
  bool _loading = false;
  String? _resultText;
  bool _isError = false;

  // The 8 inputs the model expects, with the same ranges enforced by the API.
  final List<InputField> _fields = [
    InputField('adult_mortality', 'Adult Mortality (per 1000)', 0, 1000,
        'Adult deaths per 1000 population'),
    InputField('bmi', 'Average BMI', 1, 80, 'Body Mass Index of population'),
    InputField('hiv_aids', 'HIV/AIDS (deaths per 1000)', 0.1, 60,
        'Deaths per 1000 live births (0-4 yrs)'),
    InputField('gdp', 'GDP per capita (USD)', 0, 200000, 'Gross Domestic Product'),
    InputField('income_composition', 'Income Composition (0-1)', 0, 1,
        'HDI income index'),
    InputField('schooling', 'Schooling (years)', 0, 25, 'Average years of schooling'),
    InputField('diphtheria', 'Diphtheria Immunization (%)', 0, 100,
        'Coverage among 1-year-olds'),
    InputField('thinness_1_19', 'Thinness age 1-19 (%)', 0, 50,
        'Prevalence of thinness'),
  ];

  @override
  void dispose() {
    for (final f in _fields) {
      f.controller.dispose();
    }
    super.dispose();
  }

  String? _validate(InputField f, String? value) {
    if (value == null || value.trim().isEmpty) {
      return 'Required';
    }
    final v = double.tryParse(value.trim());
    if (v == null) return 'Enter a valid number';
    if (v < f.min || v > f.max) {
      return 'Must be between ${f.min} and ${f.max}';
    }
    return null;
  }

  Future<void> _predict() async {
    // Client-side validation first (matches API constraints).
    if (!_formKey.currentState!.validate()) {
      setState(() {
        _isError = true;
        _resultText = 'Please fix the highlighted fields before predicting.';
      });
      return;
    }

    setState(() {
      _loading = true;
      _resultText = null;
      _isError = false;
    });

    final body = <String, dynamic>{};
    for (final f in _fields) {
      body[f.key] = double.parse(f.controller.text.trim());
    }

    try {
      final resp = await http
          .post(
            Uri.parse('$kApiBaseUrl/predict'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 60));

      if (resp.statusCode == 200) {
        final data = jsonDecode(resp.body);
        setState(() {
          _isError = false;
          _resultText =
              '${data['predicted_life_expectancy']} ${data['unit']}';
        });
      } else if (resp.statusCode == 422) {
        // Validation error from the API (out-of-range / wrong type / missing).
        final data = jsonDecode(resp.body);
        String msg = 'Invalid input.';
        if (data['detail'] is List && data['detail'].isNotEmpty) {
          final d = data['detail'][0];
          final loc = (d['loc'] as List).last;
          msg = 'Invalid "$loc": ${d['msg']}';
        }
        setState(() {
          _isError = true;
          _resultText = msg;
        });
      } else {
        setState(() {
          _isError = true;
          _resultText = 'Server error (${resp.statusCode}). Please try again.';
        });
      }
    } catch (e) {
      setState(() {
        _isError = true;
        _resultText =
            'Could not reach the API. Check your connection / URL.\n($e)';
      });
    } finally {
      setState(() => _loading = false);
    }
  }

  void _clear() {
    for (final f in _fields) {
      f.controller.clear();
    }
    setState(() {
      _resultText = null;
      _isError = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Life Expectancy Predictor'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 560),
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const SizedBox(height: 4),
                Text(
                  'Predict a country\'s average life expectancy from WHO health & socio-economic indicators.',
                  style: Theme.of(context).textTheme.bodyMedium,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                Card(
                  elevation: 2,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Form(
                      key: _formKey,
                      child: Column(
                        children: [
                          for (final f in _fields) ...[
                            TextFormField(
                              controller: f.controller,
                              keyboardType:
                                  const TextInputType.numberWithOptions(
                                      decimal: true, signed: false),
                              inputFormatters: [
                                FilteringTextInputFormatter.allow(
                                    RegExp(r'[0-9.]')),
                              ],
                              decoration: InputDecoration(
                                labelText: f.label,
                                helperText:
                                    '${f.hint}  (range ${f.min}–${f.max})',
                              ),
                              validator: (v) => _validate(f, v),
                            ),
                            const SizedBox(height: 14),
                          ],
                          Row(
                            children: [
                              Expanded(
                                child: FilledButton.icon(
                                  onPressed: _loading ? null : _predict,
                                  icon: _loading
                                      ? const SizedBox(
                                          width: 18,
                                          height: 18,
                                          child: CircularProgressIndicator(
                                              strokeWidth: 2,
                                              color: Colors.white),
                                        )
                                      : const Icon(Icons.analytics_outlined),
                                  label: Text(_loading
                                      ? 'Predicting...'
                                      : 'Predict'),
                                  style: FilledButton.styleFrom(
                                    padding: const EdgeInsets.symmetric(
                                        vertical: 16),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 12),
                              OutlinedButton(
                                onPressed: _loading ? null : _clear,
                                style: OutlinedButton.styleFrom(
                                  padding:
                                      const EdgeInsets.symmetric(vertical: 16),
                                ),
                                child: const Text('Clear'),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                if (_resultText != null) _buildResultCard(),
                const SizedBox(height: 24),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    final Color bg = _isError
        ? Colors.red.shade50
        : Colors.green.shade50;
    final Color fg = _isError ? Colors.red.shade800 : Colors.green.shade800;
    return Card(
      color: bg,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: fg.withOpacity(0.3)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Icon(
              _isError ? Icons.error_outline : Icons.favorite_outline,
              color: fg,
              size: 34,
            ),
            const SizedBox(height: 8),
            Text(
              _isError ? 'Error' : 'Predicted Life Expectancy',
              style: TextStyle(color: fg, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 6),
            Text(
              _resultText!,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: fg,
                fontSize: _isError ? 15 : 28,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
