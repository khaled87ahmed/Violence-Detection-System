import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:modal_progress_hud_nsn/modal_progress_hud_nsn.dart';
import 'package:security_app/constant.dart';
import 'package:security_app/features/Auth/busines_logic/auth_cubit/cubit/auth_cubit.dart';
import 'package:security_app/features/Auth/presentation/widgets/custom_button.dart';
import 'package:security_app/features/Auth/presentation/widgets/custom_textfield.dart';
import 'package:security_app/features/Auth/presentation/widgets/snackbarmessage.dart';

class Registerview extends StatelessWidget {
  bool showSpinner = false;
  static String id = 'registerView';
  GlobalKey<FormState> formKey = GlobalKey();
  String? email;
  String? password;

  @override
  Widget build(BuildContext context) {
    return BlocConsumer<AuthCubit, AuthState>(
      listener: (context, state) {
        if (state is RegisterLoading) {
          showSpinner = true;
        } else if (state is RegisterSuccess) {
          showSpinner = false;
          snackbarMessage(context, 'Register success');
          Navigator.pop(context);
        } else if (state is RegisterFailure) {
          showSpinner = false;
          snackbarMessage(context, state.errorMessage);
        }
      },
      builder: (context, state) {
        return ModalProgressHUD(
          inAsyncCall: showSpinner,
          child: Scaffold(
            backgroundColor: kPrimaryColor,
            body: SafeArea(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  return SingleChildScrollView(
                    reverse: true,
                    child: ConstrainedBox(
                      constraints:
                          BoxConstraints(minHeight: constraints.maxHeight),
                      child: IntrinsicHeight(
                        child: Form(
                          key: formKey,
                          child: Padding(
                            padding: const EdgeInsets.all(16.0),
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Spacer(),
                                Icon(Icons.security,
                                    size: 100, color: Colors.white),
                                Text('Vioguard',
                                    style: TextStyle(
                                        fontSize: 35, color: Colors.white)),
                                Row(
                                  children: [
                                    Text('Register',
                                        style: TextStyle(
                                            fontSize: 35, color: Colors.white)),
                                  ],
                                ),
                                SizedBox(height: 20),
                                custom_textField(
                                  hint: 'Email',
                                  email: email,
                                  onChanged: (value) {
                                    email = value;
                                  },
                                ),
                                SizedBox(height: 20),
                                custom_textField(
                                  hint: 'Password',
                                  onChanged: (value) {
                                    password = value;
                                  },
                                  password: password,
                                ),
                                SizedBox(height: 20),
                                CustomButton(
                                  buttonName: 'Register',
                                  onPressed: () {
                                    if (formKey.currentState!.validate()) {
                                      BlocProvider.of<AuthCubit>(context)
                                          .registerUser(
                                              email: email!,
                                              password: password!);
                                    }
                                  },
                                ),
                                SizedBox(height: 20),
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Text('Already have an account?',
                                        style: TextStyle(color: Colors.white)),
                                    TextButton(
                                      onPressed: () {
                                        Navigator.pop(context);
                                      },
                                      child: Text('Login',
                                          style:
                                              TextStyle(color: Colors.white)),
                                    ),
                                  ],
                                ),
                                Spacer(),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
        );
      },
    );
  }
}
