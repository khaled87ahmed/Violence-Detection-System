import 'package:flutter/material.dart';

class custom_textField extends StatelessWidget {
  final String hint;
  String? email;
  String? password;
  Function(String) onChanged;
  bool obsecure;

  custom_textField(
      {super.key,
      required this.hint,
      this.email,
      this.password,
      required this.onChanged,
      this.obsecure = false});
  @override
  Widget build(BuildContext context) {
    return TextFormField(
      obscureText: obsecure,
      validator: (value) {
        if (value!.isEmpty) {
          return 'Please fill the field';
        }
      },
      style: TextStyle(
        color: Colors.white,
      ),
      onChanged: onChanged,
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: TextStyle(
          color: Colors.white,
        ),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: BorderSide(
            color: Colors.white,
            width: 2,
          ),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: BorderSide(
            color: Colors.white,
            width: 2,
          ),
        ),
      ),
    );
  }
}
